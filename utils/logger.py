import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.rewards = []
        self.lengths = []

    def log(self, reward, length):
        self.rewards.append(reward)
        self.lengths.append(length)

    def plot(self):
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.plot(self.rewards)
        plt.title("Reward per Episode")

        plt.subplot(1, 2, 2)
        plt.plot(self.lengths)
        plt.title("Episode Length")

        plt.show()