from collections import deque

metrics_buffer = deque(maxlen=200)

def push_metric(data):
    metrics_buffer.append(data)

def get_metrics():
    return list(metrics_buffer)