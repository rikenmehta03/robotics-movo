import numpy as np
import gym
import random
import time

from agents import QAgent


def interact(env, agent):
    total_episodes = 50000        # Total episodes
    max_steps = 99                # Max steps per episode

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        agent.update(episode)
        for _ in range(max_steps):
            action = agent.select_action(state, env)
            new_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, new_state)
            state = new_state
            if done:
                break


def test(env, agent):
    rewards = []
    total_test_episodes = 100
    max_steps = 99

    for episode in range(total_test_episodes):
        state = env.reset()
        done = False
        total_rewards = 0

        for _ in range(max_steps):
            action = agent.select_action(state, env, False)
            new_state, reward, done, _ = env.step(action)
            total_rewards += reward
            state = new_state
            if done:
                rewards.append(total_rewards)
                print("Reward for episode {}: {}".format(episode, total_rewards))
                break

    print("average reward for total {} episodes: {}".format(
        total_test_episodes, float(sum(rewards))/total_test_episodes))


env = gym.make('Taxi-v2')
action_size = env.action_space.n
state_size = env.observation_space.n
agent = QAgent((state_size, action_size))
interact(env, agent)
test(env, agent)
