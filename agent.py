import random
import math
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

# -------------- Noisy Linear -------------- #
class NoisyLinear(nn.Module):
    """
    Factorized NoisyNet layer (Fortunato et al. 2018)
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features),
                                                    sigma_init / math.sqrt(in_features)))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init / math.sqrt(in_features)))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon = eps_out.ger(eps_in)
        self.bias_epsilon = eps_out

    def forward(self, x):
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

# -------------- Dueling + Distributional Network -------------- #
class RainbowNetwork(nn.Module):
    """
    Dueling architecture with noisy layers, distributional Q (C51).
    Input: (batch, 4, 128, 128) → 4 grayscale frames.
    Output: (batch, num_actions, num_atoms)
    """
    def __init__(self, in_channels=4, num_actions=11, num_atoms=51, hidden=256):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        test_in = torch.zeros(1, in_channels, 128, 128)
        conv_out_size = self.conv(test_in).view(1, -1).size(1)

        # Dueling streams
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_size, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, num_atoms)
        )
        self.adv_stream = nn.Sequential(
            NoisyLinear(conv_out_size, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, num_actions * num_atoms)
        )

    def forward(self, x):
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        value = self.value_stream(features)                   # (batch, num_atoms)
        adv = self.adv_stream(features).view(-1, self.num_actions, self.num_atoms)
        adv_mean = adv.mean(dim=1, keepdim=True)
        q_dist = value.unsqueeze(1) + (adv - adv_mean)
        return q_dist

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# -------------- Prioritized Replay Buffer -------------- #
class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def store(self, transition, priority=1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        total = len(self.buffer)
        prios = np.array(self.priorities) ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total_prob = probs[indices]
        weights = (total_prob * total) ** (-beta)
        weights /= weights.max()
        return samples, indices, torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, new_prios):
        for idx, prio in zip(indices, new_prios):
            self.priorities[idx] = prio.item()

# -------------- Rainbow DQN Agent -------------- #
class RainbowDQN:
    def __init__(
        self,
        state_shape=(4, 128, 128),
        num_actions=11,
        num_atoms=51,
        v_min=-2.0,
        v_max=6.0,
        gamma=0.99,
        lr=1e-4,
        buffer_size=50000,
        batch_size=32,
        beta_start=0.4,
        beta_frames=100000,
        target_update_interval=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.6)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_increment_per_frame = (1.0 - beta_start) / float(beta_frames)
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)

        self.online_net = RainbowNetwork(in_channels=state_shape[0], num_actions=num_actions, num_atoms=num_atoms).to(device)
        self.target_net = RainbowNetwork(in_channels=state_shape[0], num_actions=num_actions, num_atoms=num_atoms).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.frame_count = 0
        self.updates_done = 0
        self.target_update_interval = target_update_interval

    def select_action(self, observation):
        """
        Expects observation as a tensor of shape (4,128,128).
        """
        self.online_net.eval()
        with torch.no_grad():
            x = observation.unsqueeze(0).to(self.device)  # (1,4,128,128)
            dist = self.online_net(x)  # (1, num_actions, num_atoms)
            dist = torch.softmax(dist, dim=2)
            q_values = torch.sum(dist * self.support.view(1,1,-1), dim=2)
            action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, transition):
        # transition: (obs, action, reward, next_obs, done)
        self.buffer.store(transition)

    def update(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        self.updates_done += 1
        beta = min(1.0, self.beta_start + self.beta_increment_per_frame * self.frame_count)
        samples, indices, weights = self.buffer.sample(self.batch_size, beta=beta)
        weights = weights.to(self.device)

        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
        obs_batch_t = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in obs_batch], dim=0).to(self.device)
        next_obs_batch_t = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in next_obs_batch], dim=0).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        dist = self.online_net(obs_batch_t)
        dist = torch.log_softmax(dist, dim=2)
        dist_action = dist.gather(1, actions_t.unsqueeze(2).expand(self.batch_size, 1, self.num_atoms)).squeeze(1)

        with torch.no_grad():
            next_dist = self.online_net(next_obs_batch_t)
            next_dist = torch.softmax(next_dist, dim=2)
            q_values_next = torch.sum(next_dist * self.support.view(1, 1, -1), dim=2)
            best_actions = q_values_next.argmax(dim=1)
            target_dist = self.target_net(next_obs_batch_t)
            target_dist = torch.softmax(target_dist, dim=2)
            target_dist = target_dist[range(self.batch_size), best_actions]

        Tz = rewards_t + (1.0 - dones_t) * self.gamma * self.support.view(1, -1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).to(self.device)
        target_dist_projected = torch.zeros_like(target_dist)
        target_dist_projected.view(-1).index_add_(0, (l + offset).view(-1),
                                                   (target_dist * (u.float() - b)).view(-1))
        target_dist_projected.view(-1).index_add_(0, (u + offset).view(-1),
                                                   (target_dist * (b - l.float())).view(-1))

        losses = -torch.sum(target_dist_projected * dist_action, dim=1)
        weighted_losses = losses * weights.squeeze()
        loss = weighted_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = weighted_losses.detach().abs().cpu() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        if self.updates_done % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        if self.updates_done % 100 == 0:
            print(f"Update {self.updates_done}: Loss = {loss.item():.4f}")


