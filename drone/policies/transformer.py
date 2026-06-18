"""
drone.policies.transformer — Transformer-based feature extractor for LIDAR observations.

Architecture:
  obs (74,)
    ├─ LIDAR beams [0:N_LIDAR_BEAMS]
    │    Each beam i → [dist_i, cos(2πi/N), sin(2πi/N)]  (angle-encoded, fixed)
    │    Linear(3, D_MODEL) → sequence of (N_LIDAR_BEAMS, D_MODEL) tokens
    │
    └─ Context features [N_LIDAR_BEAMS:]
         Linear(context_dim, D_MODEL) → one context token prepended as [CLS]
                                               │
                         Sequence: (1 + N_LIDAR_BEAMS, D_MODEL)
                                               │
                     TransformerEncoder(layers=N_LAYERS, heads=N_HEADS, d=D_MODEL)
                                               │
                           Mean-pool all tokens → (D_MODEL,)
                                               │
                           Linear(D_MODEL, FEATURES_DIM) + LayerNorm + ReLU
                                               │
                                      features (FEATURES_DIM,)

SB3 Usage (PPO or SAC — works with any algorithm):

    from drone.policies.transformer import make_transformer_kwargs

    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=make_transformer_kwargs(),
    )
    model = SAC(
        "MlpPolicy", env,
        policy_kwargs=make_transformer_kwargs(features_dim=128),
    )

Design notes:
- Angle encoding is FIXED (sin/cos of beam angle) — the model never learns to confuse
  north with south because the geometric embedding is baked in.
- The [CLS] context token (goal, velocity, wind, motor status) attends to ALL LIDAR
  beams via self-attention, so the policy implicitly learns "which beams matter given
  my current goal direction and energy."
- Mean-pooling is more stable than CLS-only during early training.
- Works with obs_dim >= N_LIDAR_BEAMS; gracefully handles the 10-dim legacy LIDAR obs.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---------------------------------------------------------------------------
# CONFIG — every architectural constant lives here
# ---------------------------------------------------------------------------
N_LIDAR_BEAMS: int = 64        # must match DroneEnvAdvanced
D_MODEL: int = 64               # transformer embedding dimension
N_HEADS: int = 4                # attention heads  (D_MODEL must be divisible)
N_LAYERS: int = 2               # transformer encoder layers
D_FF: int = 128                 # feed-forward hidden dim inside each layer
DROPOUT: float = 0.0            # 0 for RL (small batches — dropout hurts)
FEATURES_DIM: int = 128         # output feature dimension fed to SB3 policy/value heads


# ---------------------------------------------------------------------------
# BEAM ANGLE ENCODING
# ---------------------------------------------------------------------------

def _make_angle_encoding(n_beams: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) tensors of shape (n_beams,) for uniform 360° scan."""
    angles = torch.tensor(
        [2.0 * math.pi * i / n_beams for i in range(n_beams)],
        dtype=torch.float32,
    )
    return torch.cos(angles), torch.sin(angles)


# ---------------------------------------------------------------------------
# TRANSFORMER LAYER (hand-rolled for transparency — no hidden magic)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single pre-LN transformer layer: MHA → FF, with pre-layer normalisation."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def __repr__(self) -> str:
        return (
            f"TransformerEncoderLayer(d={self.norm1.normalized_shape[0]}, "
            f"heads={self.self_attn.num_heads})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN attention
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        x = x + self.drop(attn_out)
        # Pre-LN feed-forward
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# FEATURE EXTRACTOR (plugs into SB3 via policy_kwargs)
# ---------------------------------------------------------------------------

class LidarTransformerExtractor(BaseFeaturesExtractor):
    """
    SB3-compatible Transformer feature extractor for LIDAR-based observations.

    Splits the observation into LIDAR beams and context features, processes
    them through a Transformer encoder, and returns a flat feature vector
    for SB3's policy/value MLP heads.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = FEATURES_DIM,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__(observation_space, features_dim)

        obs_dim: int = int(np.prod(observation_space.shape))
        self.n_lidar: int = min(N_LIDAR_BEAMS, obs_dim)
        self.context_dim: int = obs_dim - self.n_lidar
        self.d_model: int = d_model

        # Fixed angle encoding — registered as buffer so it moves to GPU automatically
        cos_enc, sin_enc = _make_angle_encoding(self.n_lidar)
        self.register_buffer("_beam_cos", cos_enc)   # (n_lidar,)
        self.register_buffer("_beam_sin", sin_enc)   # (n_lidar,)

        # Beam projection: [dist, cos, sin] → d_model
        self.beam_proj = nn.Linear(3, d_model)

        # Context token projection (or zero-token if no context features)
        if self.context_dim > 0:
            self.ctx_proj = nn.Linear(self.context_dim, d_model)
        else:
            self.ctx_proj = None

        # Transformer encoder
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # Projection to SB3 features_dim
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

        # Weight init
        self._init_weights()

    def __repr__(self) -> str:
        return (
            f"LidarTransformerExtractor("
            f"n_lidar={self.n_lidar}, ctx={self.context_dim}, "
            f"d_model={self.d_model}, layers={len(self.encoder)}, "
            f"features_dim={self.features_dim})"
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, obs_dim) float tensor from SB3
        returns: (batch, features_dim)
        """
        B = obs.shape[0]

        # Split observation
        lidar = obs[:, : self.n_lidar]              # (B, n_lidar)
        context = obs[:, self.n_lidar :]            # (B, context_dim)

        # Build per-beam tokens: [dist, cos_angle, sin_angle]
        cos_enc = self._beam_cos.unsqueeze(0).expand(B, -1)   # (B, n_lidar)
        sin_enc = self._beam_sin.unsqueeze(0).expand(B, -1)   # (B, n_lidar)
        beam_feats = torch.stack([lidar, cos_enc, sin_enc], dim=2)  # (B, n_lidar, 3)
        beam_tokens = self.beam_proj(beam_feats)               # (B, n_lidar, d_model)

        # Build sequence: [CLS_context | beam_0 | beam_1 | ... | beam_63]
        if self.ctx_proj is not None and self.context_dim > 0:
            cls_token = self.ctx_proj(context).unsqueeze(1)    # (B, 1, d_model)
            sequence = torch.cat([cls_token, beam_tokens], dim=1)  # (B, 1+n_lidar, d_model)
        else:
            sequence = beam_tokens                              # (B, n_lidar, d_model)

        # Transformer encoder
        x = sequence
        for layer in self.encoder:
            x = layer(x)
        x = self.encoder_norm(x)

        # Mean-pool over sequence dimension → (B, d_model)
        pooled = x.mean(dim=1)

        return self.out_proj(pooled)                           # (B, features_dim)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# FACTORY FUNCTION — the public API most users will call
# ---------------------------------------------------------------------------

def make_transformer_kwargs(
    features_dim: int = FEATURES_DIM,
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    d_ff: int = D_FF,
    net_arch: Optional[list] = None,
) -> Dict:
    """
    Return policy_kwargs dict for use with any SB3 algorithm.

    Example:
        model = PPO("MlpPolicy", env, policy_kwargs=make_transformer_kwargs())
        model = SAC("MlpPolicy", env, policy_kwargs=make_transformer_kwargs(features_dim=128))
    """
    return {
        "features_extractor_class": LidarTransformerExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ff": d_ff,
        },
        "net_arch": net_arch if net_arch is not None else [64],
    }


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces

    print("=== LidarTransformerExtractor smoke test ===")

    # Simulate DroneEnvAdvanced obs space (74,)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(74,), dtype=np.float32)
    extractor = LidarTransformerExtractor(obs_space)
    print(repr(extractor))
    print(f"Parameters: {extractor.count_parameters():,}")

    # Forward pass
    batch = torch.zeros(4, 74)
    out = extractor(batch)
    print(f"Input:  {batch.shape}")
    print(f"Output: {out.shape}  (expected: [4, {FEATURES_DIM}])")
    assert out.shape == (4, FEATURES_DIM), f"Wrong output shape: {out.shape}"

    # Test with 10-dim legacy obs (DroneEnvLidar fallback)
    obs_space_10 = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    ext_small = LidarTransformerExtractor(obs_space_10, features_dim=64)
    print(f"\nLegacy 10-dim obs extractor: {repr(ext_small)}")
    out_small = ext_small(torch.zeros(2, 10))
    print(f"Output: {out_small.shape}  (expected: [2, 64])")
    assert out_small.shape == (2, 64)

    # Policy kwargs
    kwargs = make_transformer_kwargs()
    print(f"\nmake_transformer_kwargs() keys: {list(kwargs.keys())}")

    print("\nAll OK")
