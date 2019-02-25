import numpy as np
import torch
import gym
import argparse
import os
import math
from timeit import default_timer as timer

import torch
import torch.nn as nn

import movo_bot

import data
import logger
import utils
from td3 import TD3, StateTransformer


def create_policy(policy_name, device, state_dim, action_dim, max_action):
    if policy_name == 'TD3':
        return TD3(device, state_dim, action_dim, max_action)
    assert 'Unknown policy: %s' % policy_name


def evaluate_policy(env, policy, tracker, num_episodes=10):
    tracker.reset('eval_episode_reward')
    tracker.reset('eval_episode_timesteps')
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        sum_reward = 0
        timesteps = 0
        while not done:
            with torch.no_grad():
                with utils.eval_mode(policy):
                    action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            sum_reward += reward
            timesteps += 1

        tracker.update('eval_episode_reward', sum_reward)
        tracker.update('eval_episode_timesteps', timesteps)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_name', default='TD3')  # Policy name
    parser.add_argument('--env_name', default='movobot-v0')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_timesteps', default=10000, type=int)
    parser.add_argument('--eval_freq', default=5e3, type=int)
    parser.add_argument('--max_timesteps', default=1e6, type=int)
    parser.add_argument('--expl_noise', default=0.1, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--log_format', default='text', type=str)
    parser.add_argument('--save_model_dir', default=None, type=str)
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--eval', default='false', type=str)
    parser.add_argument('--discrete_reward', default=False, action='store_true')
    parser.add_argument('--model_dir', default='models', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    env = gym.make(args.env_name)

    if args.env_name == 'movobot-v0':
        env.init(render=args.render, discrete_reward=args.discrete_reward)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create replay buffers
    replay_buffer = data.ReplayBuffer()

    # pylint: disable=E1101
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pylint: enable=E1101
    
    # Initialize policy
    train_policy = create_policy(
        args.policy_name, device, state_dim, action_dim, max_action)

    train_policy.train()

    tracker = logger.StatsTracker()
    train_logger = logger.TrainLogger(args.log_format)
    eval_logger = logger.EvalLogger(args.log_format)
    
    state_transformer = StateTransformer()

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = False
    state = state_transformer.transform(env.reset())

    
    if args.eval == 'true':
        train_policy.load(args.model_dir)
        print('Entering evaluation mode')
    else:
        while total_timesteps < args.max_timesteps:
            if done:
                if total_timesteps != 0:
                    train_logger.dump(tracker)
                    if args.policy_name == 'TD3':
                        start = timer()
                        train_policy.run(replay_buffer, episode_timesteps, tracker,
                                        args.batch_size, args.discount, args.tau,
                                        args.policy_noise, args.noise_clip,
                                        args.policy_freq)
                        print('run: %.3fs' % (timer() - start))
                    # else:
                    #     train_policy.train(replay_buffer, episode_timesteps,
                    #                        args.batch_size, args.discount, args.tau)

                # Evaluate episode
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    evaluate_policy(env, train_policy, tracker)
                    eval_logger.dump(tracker)
                    tracker.reset('train_episode_reward')
                    tracker.reset('train_episode_timesteps')
                    if args.save_model_dir is not None:
                        train_policy.save(args.save_model_dir)
                print("episode_reward : {}".format(episode_reward))
                tracker.update('train_episode_reward', episode_reward)
                tracker.update('train_episode_timesteps', episode_timesteps)
                # Reset environment
                state = state_transformer.transform(env.reset())
                done = False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                tracker.update('num_episodes')
                tracker.reset('episode_timesteps')

            # Select action randomly or according to policy
            if total_timesteps < args.start_timesteps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    with utils.eval_mode(train_policy):
                        action = train_policy.select_action(np.array(state))
                if args.expl_noise != 0:
                    action = (action + np.random.normal(
                        0, args.expl_noise, size=env.action_space.shape[0])).clip(
                            env.action_space.low, env.action_space.high)

            # Perform action
            start = timer()
            new_state, reward, done, _ = env.step(action)
            print('step: %.3fs' % (timer() - start))
            new_state = state_transformer.transform(new_state)
            done_float = 0 if episode_timesteps + \
                1 == env._max_episode_steps else float(done)
            episode_reward += reward

            # Store data in replay buffer
            replay_buffer.add((state, new_state, action, reward, done_float))

            state = new_state

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
            tracker.update('total_timesteps')
            tracker.update('episode_timesteps')

    # Final evaluation
    
    evaluate_policy(env, train_policy, tracker)
    eval_logger.dump(tracker)
    if args.save_model_dir is not None:
        train_policy.save(args.save_model_dir)


if __name__ == '__main__':
    main()
