import gym
from baselines import deepq
import balance_bot
 
def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved
 
def main():
    # create the environment
    env = gym.make("balancebot-v0") # <-- this we need to create
 
    # create the learning agent
 
    # train the agent on the environment
    act = deepq.learn(
        env, network='mlp', lr=1e-3,
        total_timesteps=200000, buffer_size=50000, exploration_fraction=0.1,
        exploration_final_eps=0.02, print_freq=10, callback=callback,load_path="balance.pkl"
    )
 
    # save trained model
    act.save("balance.pkl")
 
if __name__ == '__main__':
    main()