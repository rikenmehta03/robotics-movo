import gym
import movo_bot
import sys
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

n_steps = 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps
    # Print stats every 1000 calls
    n_steps += 128
    if (n_steps) % 1024 == 0:
        _locals['self'].save('movo.pkl')
    
    return True

env = gym.make("movobot-v0")
render = True

if len(sys.argv) > 1:
    render = bool(int(sys.argv[1]))

env.init(render=render)
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1, n_steps=128)
model.learn(total_timesteps=100000, callback=callback)

model.save("movo.pkl")

