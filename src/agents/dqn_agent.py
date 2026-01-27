import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
import numpy as np
from collections import deque
from src.utils import config

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        # Fixed: Changed optimizers reference
        self.optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)

    def _build_model(self):
        # Fixed: Correctly using layers and models aliases
        model = models.Sequential([
            layers.Input(shape=(config.STATE_W, config.STATE_H, config.STACK_SIZE)),
            layers.Rescaling(1./255), # Normalize here instead of during training for efficiency
            layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
            layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
            layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(config.ACTION_SPACE, activation='linear')
        ])
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(config.ACTION_SPACE)
        # state is expected to be (84, 84, 4)
        q_values = self.model.predict(state[np.newaxis, ...], verbose=0)
        return np.argmax(q_values[0])

    @tf.function # Boosts performance by compiling the training step
    def train_step(self, states, actions, y_targets):
        with tf.GradientTape() as tape:
            current_q = self.model(states, training=True)
            one_hot_actions = tf.one_hot(actions, config.ACTION_SPACE)
            selected_action_q = tf.reduce_sum(current_q * one_hot_actions, axis=1)
            loss = tf.keras.losses.Huber()(y_targets, selected_action_q)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def train(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        batch = random.sample(self.memory, config.BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Optimization: Rescaling is now handled in the model layers
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Double DQN Logic
        next_q_online = self.model.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_online, axis=1)
        
        next_q_target = self.target_model.predict(next_states, verbose=0)
        target_values = next_q_target[np.arange(config.BATCH_SIZE), next_actions]

        y_targets = rewards + (1 - dones) * config.GAMMA * target_values
        
        return self.train_step(states, actions, y_targets)