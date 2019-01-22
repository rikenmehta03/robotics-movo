import gym
import balance_bot
from baselines import deepq


def main():
    env = gym.make("balancebot-v0")
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="balance.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()