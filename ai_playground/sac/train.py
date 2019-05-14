import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time

sys.path.append('../')

from jaco import jaco_reacher

from logger import Logger
from video import VideoRecorder
import utils
from sac import SAC

_N_SUB_STEPS = 100

class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._max_episode_steps = _N_SUB_STEPS
        if hasattr(env.action_space, 'dtype'):
            dtype = env.action_space.dtype
        else:
            dtype = env.action_space.sample().dtype

        self.action_space = gym.spaces.Box(
            -1,
            1,
            shape=self.env.action_space.shape,
            dtype=dtype)

    def action(self, a):
        l = self.env.action_space.low
        h = self.env.action_space.high

        a = l + (a + 1.0) * 0.5 * (h - l)
        a = np.clip(a, l, h)
        return a

    def reverse_action(self, a):
        l = self.env.action_space.low
        h = self.env.action_space.high

        a = 2 * (a - l) / (h - l) - 1
        a = np.clip(a, l, h)

        return a


def evaluate_policy(env, policy, step, L, num_episodes, video_dir=None):
    for i in range(num_episodes):
        video = VideoRecorder(env, enabled=video_dir is not None and i == 0, fps=15)
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                with utils.eval_mode(policy):
                    action = policy.select_action(obs)

            obs, reward, done, _ = env.step(action)
            video.record()
            total_reward += reward

        video.save(video_dir, '%d.mp4' % step)
        L.log('eval/episode_reward', total_reward, step)
    L.dump(step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_name', default='point_mass')
    parser.add_argument('--task_name', default='easy')
    parser.add_argument('--start_steps', default=1000, type=int)
    parser.add_argument('--eval_freq', default=5000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument('--max_steps', default=1e6, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--replay_buffer_size', default=1000000, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--initial_temperature', default=0.01, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--policy_freq', default=2, type=int)
    parser.add_argument('--save_dir', default='.', type=str)
    parser.add_argument('--no_save', default=False, action='store_true')
    parser.add_argument('--no_render', default=False, action='store_true')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--use_tb', default=False, action='store_true')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    utils.set_seed_everywhere(args.seed)
    env = NormalizedActions(jaco_reacher.env)

    utils.make_dir(args.save_dir)
    model_dir = None if args.no_save else utils.make_dir(
        os.path.join(args.save_dir, 'model'))
    video_dir = None if args.no_render else utils.make_dir(
        os.path.join(args.save_dir, 'video'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    replay_buffer = utils.ReplayBuffer(args.replay_buffer_size)
    L = Logger(args.save_dir, use_tb=args.use_tb)

    policy = SAC(device, state_dim, action_dim, args.initial_temperature,
                 args.lr)

    step = 0
    steps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_step = 0
    done = True

    evaluate_policy(env, policy, step, L, args.num_eval_episodes, video_dir)

    start_time = time.time()
    while step < args.max_steps:

        if done:
            if step != 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate episode
            if steps_since_eval >= args.eval_freq:
                steps_since_eval %= args.eval_freq
                evaluate_policy(env, policy, step, L, args.num_eval_episodes,
                                video_dir)

                if model_dir is not None:
                    policy.save(model_dir, step)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode_num += 1

            L.log('train/episode', episode_num, step)

        # Select action randomly or according to policy
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                with utils.eval_mode(policy):
                    action = policy.sample_action(obs)

        if step >= args.start_steps:
            num_updates = args.start_steps if step == args.start_steps else 1
            for _ in range(num_updates):
                policy.update(
                    replay_buffer,
                    step,
                    L,
                    args.batch_size,
                    args.discount,
                    args.tau,
                    args.policy_freq,
                    target_entropy=-action_dim)

        next_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done)
        episode_reward += reward

        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs

        episode_step += 1
        step += 1
        steps_since_eval += 1


if __name__ == '__main__':
    main()
