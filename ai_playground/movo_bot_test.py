import gym
import movo_bot
import os
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import argparse

n_steps = 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='movobot-v0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eval_freq', default=5e3, type=int)
    parser.add_argument('--max_timesteps', default=1e5, type=int)
    parser.add_argument('--save_file_name', default='movo.pkl', type=str)
    parser.add_argument('--render', default=1, type=int)
    parser.add_argument('--eval', default='false', type=str)
    parser.add_argument('--model_dir', default='models', type=str)
    parser.add_argument('--discrete', default=1, type=int)
    args = parser.parse_args()
    return args
    
args = parse_args()

model_file = os.path.join(args.model_dir, args.save_file_name)

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps
    
    n_steps += 128
    if (n_steps) % 1024 == 0:
        _locals['self'].save(model_file)
    
    return True




env = gym.make("movobot-v0")
env.init(render=(args.render==1), isDiscrete=(args.discrete==1))
env = DummyVecEnv([lambda: env])

if args.eval == 'false':
    model = PPO2(MlpPolicy, env, verbose=1, n_steps=128)
    model.learn(total_timesteps=100000, callback=callback)
    model.save(model_file)
else:
    model = PPO2.load(model_file)
    for i in range(10):
        obs = env.reset()
        sum_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            sum_reward += rewards
            steps += 1
        print("Avg reward: {}".format(sum_reward/steps))
        