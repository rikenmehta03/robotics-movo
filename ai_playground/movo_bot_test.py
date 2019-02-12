import gym
import movo_bot
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


env = gym.make("movobot-v0")
env.init()
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)


model.save("movo.pkl")
# state = env.reset()
# print(state.shape)

