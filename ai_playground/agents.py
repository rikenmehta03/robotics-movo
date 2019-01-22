import numpy as np
import random

class QAgent():
    def __init__(self, q_size):
        self.Q = np.zeros(q_size)
    
        self.gamma = 0.618
        self.alpha = 0.7

        self.epsilon = 1.0
        self.max_epsilon = 1.0
        self.min_epsilon = 0.01
        self.decay_rate = 0.01
    
    def select_action(self, state, env, experiment=True):
        if experiment:
            trade_off = random.uniform(0,1)
            if trade_off > self.epsilon:
                action = np.argmax(self.Q[state, :])
            else:
                action = env.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        
        return action
    
    def step(self, state, action, reward, new_state):
        self.Q[state, action] += self.alpha * (reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action])
    
    def update(self, episode):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon)*np.exp(-self.decay_rate*episode)
        