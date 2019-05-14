import torch
import numpy as np
import torch.nn as nn
import os
import random


class eval_mode(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        self.prev = self.model.training
        self.model.train(False)

    def __exit__(self, *args):
        self.model.train(self.prev)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class ReplayBuffer(object):
    def __init__(self, capacity=100000):
        self._storage = []
        self._capacity = capacity
        self._idx = 0

    def add(self, obs, action, reward, next_obs, done):
        data = (obs, action, reward, next_obs, done)
        if len(self._storage) == self._capacity:
            self._storage[self._idx] = data
            self._idx = (self._idx + 1) % self._capacity
        else:
            self._storage.append(data)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self._storage), size=batch_size)
        obs, action, reward, next_obs, done = [], [], [], [], []

        for i in idxs:
            data = self._storage[i]
            obs.append(np.array(data[0], copy=False))
            action.append(np.array(data[1], copy=False))
            reward.append(np.array(data[2], copy=False))
            next_obs.append(np.array(data[3], copy=False))
            done.append(np.array(data[4], copy=False))

        obs = np.array(obs)
        action = np.array(action)
        reward = np.array(reward).reshape(-1, 1)
        next_obs = np.array(next_obs)
        done = np.array(done).reshape(-1, 1)

        return obs, action, reward, next_obs, done
