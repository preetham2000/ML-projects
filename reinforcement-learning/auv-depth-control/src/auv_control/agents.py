"""Hand-written DDPG and TD3 agents from the original experiment."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-capacity replay memory for continuing-task transitions."""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """State-action value network with two 128-unit hidden layers."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


class PolicyNetwork(nn.Module):
    """Bounded deterministic policy with two 128-unit hidden layers."""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)


class DDPGAgent:
    """DDPG actor-critic with soft targets and optional Gaussian exploration."""

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = PolicyNetwork(state_dim, action_dim, max_action).to(DEVICE)
        self.critic = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_actor = PolicyNetwork(state_dim, action_dim, max_action).to(DEVICE)
        self.target_critic = QNetwork(state_dim, action_dim).to(DEVICE)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay = ReplayBuffer()
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, noise_scale=0.0):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = self.actor(state_tensor).cpu().detach().numpy()[0]
        if noise_scale > 0:
            action += np.random.normal(scale=noise_scale, size=action.shape)
        return action

    def train(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        states, actions, rewards, next_states = self.replay.sample(batch_size)
        state = torch.FloatTensor(states).to(DEVICE)
        action = torch.FloatTensor(actions).to(DEVICE)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_states).to(DEVICE)

        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target = reward + self.gamma * target_q

        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for parameter, target_parameter in zip(
            self.actor.parameters(), self.target_actor.parameters()
        ):
            target_parameter.data.copy_(
                self.tau * parameter.data + (1 - self.tau) * target_parameter.data
            )
        for parameter, target_parameter in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_parameter.data.copy_(
                self.tau * parameter.data + (1 - self.tau) * target_parameter.data
            )


class TD3Agent:
    """TD3 agent with twin critics, target smoothing, and delayed policy updates."""

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        self.actor = PolicyNetwork(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = PolicyNetwork(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.critic1_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = QNetwork(state_dim, action_dim).to(DEVICE)
        self.critic2_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=1e-3)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=1e-3)

        self.replay = ReplayBuffer()
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        return self.actor(state_tensor).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        self.total_it += 1
        states, actions, rewards, next_states = self.replay.sample(batch_size)
        state = torch.FloatTensor(states).to(DEVICE)
        action = torch.FloatTensor(actions).to(DEVICE)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)
        next_state = torch.FloatTensor(next_states).to(DEVICE)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target)
        critic2_loss = nn.MSELoss()(current_q2, target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for parameter, target_parameter in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_parameter.data.copy_(
                    self.tau * parameter.data
                    + (1 - self.tau) * target_parameter.data
                )
            for critic, critic_target in (
                (self.critic1, self.critic1_target),
                (self.critic2, self.critic2_target),
            ):
                for parameter, target_parameter in zip(
                    critic.parameters(), critic_target.parameters()
                ):
                    target_parameter.data.copy_(
                        self.tau * parameter.data
                        + (1 - self.tau) * target_parameter.data
                    )
