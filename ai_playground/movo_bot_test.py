import gym
import movo_bot
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make("movobot-v0")
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=50000)


model.save("movo.pkl")
