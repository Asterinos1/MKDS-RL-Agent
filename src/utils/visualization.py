import matplotlib.pyplot as plt
import os

class TrainVisualizer:
    def __init__(self, save_path="outputs/progress.png"):
        self.save_path = save_path
        self.episodes = []
        self.rewards = []
        self.epsilons = []
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def add_data(self, episode, reward, epsilon):
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self._plot()

    def _plot(self):
        plt.figure(figsize=(10, 5))
        #Reward
        plt.subplot(1, 2, 1)
        plt.plot(self.episodes, self.rewards, color='blue')
        plt.title('Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        #Epsilon (Exploration)
        plt.subplot(1, 2, 2)
        plt.plot(self.episodes, self.epsilons, color='red')
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close() 