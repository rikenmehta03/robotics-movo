import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.stats


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1


class TD3(object):
    def __init__(self, device, state_dim, action_dim, max_action):
        self.device = device
        self.max_action = max_action
        self.training = True

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

    def train(self, mode=True):
        def check_and_set(module):
            assert module.training == self.training
            module.train(mode)
        check_and_set(self.actor)
        check_and_set(self.actor_target)
        check_and_set(self.critic)
        check_and_set(self.critic_target)

        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def save(self, directory):
        torch.save(self.actor.state_dict(), '%s/actor.pt' % directory)
        torch.save(self.critic.state_dict(), '%s/critic.pt' % directory)

    def load(self, directory):
        self.actor.load_state_dict(torch.load('%s/actor.pt' % directory))
        self.critic.load_state_dict(torch.load('%s/critic.pt' % directory))

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def run(self, replay_buffer, num_iterations, tracker,
            batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
            policy_freq=2, predict_reward=None):

        for it in range(num_iterations):
            # Sample replay buffer
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)
            tracker.update('train_reward', reward.sum().item(), reward.size(0))
            if predict_reward is not None:
                #reward = predict_reward(state, action, next_state) * (1 - discount)
                predicted_reward = predict_reward(state, action, next_state)

                tracker.update('train_predicted_reward', predicted_reward.sum(
                ).item(), predicted_reward.size(0))
                tracker.update('reward_pearsonr', scipy.stats.pearsonr(
                    reward.cpu().numpy(), predicted_reward.cpu().numpy())[0][0])

            # Select action according to policy and add clipped noise
            noise = torch.FloatTensor(u).data.normal_(
                0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            if predict_reward is None:
                target_Q = reward + (done * discount * target_Q).detach()
            else:
                target_Q = predicted_reward + \
                    (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q)
            tracker.update('critic_loss', critic_loss *
                           target_Q.size(0), target_Q.size(0))

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                tracker.update('actor_loss', actor_loss *
                               state.size(0), state.size(0))

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(
                        self.critic.parameters(),
                        self.critic_target.parameters()):

                    target_param.data.copy_(tau * param.data +
                                            (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(),
                                               self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data +
                                            (1 - tau) * target_param.data)
