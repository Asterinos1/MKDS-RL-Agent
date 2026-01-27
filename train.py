import os
import tensorflow as tf
from env.mkds_env import MKDSEnv
from src.utils.wrappers import FrameStacker
from src.utils.visualization import TrainVisualizer
from src.agents.dqn_agent import DQNAgent
from src.utils import config

def train():
    env = MKDSEnv()
    agent = DQNAgent()
    stacker = FrameStacker()
    visualizer = TrainVisualizer()
    epsilon = config.EPSILON_START
    total_steps = 0
    print("Starting Training on Figure-8 Circuit...")

    for episode in range(1, 10001):
        #reset environment and Stack Frames
        raw_frame = env.reset()
        state = stacker.reset(raw_frame)
        episode_reward = 0
        done = False
        
        while not done:
            #agent chooses action (Epsilon-Greedy)
            action = agent.act(state, epsilon)
            #environment step (returns 84x84 grayscale frame)
            next_raw_frame, reward, done = env.step(action)
            next_state = stacker.append(next_raw_frame)
            #store experience in replay buffer
            agent.memory.append((state, action, reward, next_state, done))
            #periodic training
            if total_steps > config.BATCH_SIZE and total_steps % 4 == 0:
                agent.train()
            state = next_state
            episode_reward += reward
            total_steps += 1
            #linear epsilon decay
            if epsilon > config.EPSILON_END:
                epsilon -= (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY
        #logging
        print(f"Episode: {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")
        visualizer.add_data(episode, episode_reward, epsilon)

        if episode % 10 == 0:
            agent.update_target_network()
            agent.model.save_weights("outputs/mkds_dqn_weights.h5")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Training on GPU: {gpus[0].name}")
    train()