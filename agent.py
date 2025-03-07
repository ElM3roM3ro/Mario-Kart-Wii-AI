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
    We assume input shape: (batch, 4, 128, 128) => 4 grayscale frames.
    Output shape: (batch, num_actions, num_atoms).
    """
    def __init__(self, in_channels=4, num_actions=11, num_atoms=51, hidden=256):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Example conv to handle 128x128 -> smaller feature map
        # Adjust to your liking
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),  # -> approx 31x31
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # -> approx 14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),           # -> approx 6x6
            nn.ReLU(),
        )

        # Test how many features remain after conv
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
        features = features.view(features.size(0), -1)  # flatten

        value = self.value_stream(features)                   # => (batch, num_atoms)
        adv = self.adv_stream(features).view(-1, self.num_actions, self.num_atoms)

        # Dueling: Q = V + (A - mean(A)) for each atom
        adv_mean = adv.mean(dim=1, keepdim=True)              # => (batch,1,num_atoms)
        q_dist = value.unsqueeze(1) + (adv - adv_mean)         # => (batch, num_actions, num_atoms)
        return q_dist

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# -------------- Prioritized Replay -------------- #
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
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
        if len(self.buffer) == self.capacity:
            total = self.capacity
        else:
            total = len(self.buffer)

        prios = np.array(self.priorities[:total]) ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Importance weights
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
            state_shape=(4,128,128),
            num_actions=11,         # match environment
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            gamma=0.99,
            lr=1e-4,
            buffer_size=50000,
            batch_size=32,
            alpha=0.6,
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

        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_increment_per_frame = (1.0 - beta_start) / float(beta_frames)

        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)

        # Online & target networks
        in_channels = state_shape[0]  # 4 for (4,128,128)
        self.online_net = RainbowNetwork(in_channels, num_actions, num_atoms).to(device)
        self.target_net = RainbowNetwork(in_channels, num_actions, num_atoms).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.frame_count = 0
        self.updates_done = 0
        self.target_update_interval = target_update_interval

    def select_action(self, observation):
        """
        observation shape: (4,128,128) in grayscale
        -> convert to Tensor shape (1,4,128,128)
        -> forward pass, pick best action (argmax of expected Q)
        """
        self.online_net.eval()
        with torch.no_grad():
            obs_t = self._preprocess_observation(observation)  # => (1,4,128,128)
            dist = self.online_net(obs_t)  # => (1, num_actions, num_atoms)
            dist = torch.softmax(dist, dim=2)  # convert to prob distribution
            q_values = torch.sum(dist * self.support.view(1, 1, -1), dim=2)  # => (1, num_actions)
            action = q_values.argmax(dim=1).item()
        return action

    def _preprocess_observation(self, obs_np):
        # (4,128,128) -> Tensor (1,4,128,128)
        x = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        return x

    def store_transition(self, transition):
        # transition = (obs, action, reward, next_obs, done)
        self.buffer.store(transition)

    def update(self):
        """
        Sample from replay, compute distributional loss, do gradient step.
        """
        if len(self.buffer.buffer) < self.batch_size:
            return  # not enough data

        self.updates_done += 1
        beta = min(1.0, self.beta_start + self.beta_increment_per_frame * self.frame_count)

        samples, indices, weights = self.buffer.sample(self.batch_size, beta=beta)
        weights = weights.to(self.device)

        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)

        obs_batch_t = self._batch_obs(obs_batch)           # => (batch,4,128,128)
        next_obs_batch_t = self._batch_obs(next_obs_batch) # => (batch,4,128,128)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)   # => (batch,1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) # => (batch,1)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)     # => (batch,1)

        dist = self.online_net(obs_batch_t)       # (batch, num_actions, num_atoms)
        dist = torch.log_softmax(dist, dim=2)     # log probabilities
        dist_action = dist.gather(1, actions_t.unsqueeze(2).expand(self.batch_size, 1, self.num_atoms))
        dist_action = dist_action.squeeze(1)      # => (batch, num_atoms)

        with torch.no_grad():
            # Next state distribution
            next_dist = self.online_net(next_obs_batch_t)    # (batch, num_actions, num_atoms)
            next_dist = torch.softmax(next_dist, dim=2)
            # Argmax w.r.t. Q-values from online_net
            q_values_next = torch.sum(next_dist * self.support.view(1,1,-1), dim=2) # => (batch, num_actions)
            best_actions = q_values_next.argmax(dim=1)       # => (batch,)

            # Evaluate target_net
            target_dist = self.target_net(next_obs_batch_t)  # => (batch, num_actions, num_atoms)
            target_dist = torch.softmax(target_dist, dim=2)
            target_dist = target_dist[range(self.batch_size), best_actions]  # => (batch, num_atoms)

        # Project distribution
        Tz = rewards_t + (1.0 - dones_t) * self.gamma * self.support.view(1, -1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms,
                                self.batch_size).long().unsqueeze(1).to(self.device)
        target_dist_projected = torch.zeros_like(target_dist)
        target_dist_projected.view(-1).index_add_(
            0, (l + offset).view(-1),
            (target_dist * (u.float() - b)).view(-1)
        )
        target_dist_projected.view(-1).index_add_(
            0, (u + offset).view(-1),
            (target_dist * (b - l.float())).view(-1)
        )

        loss = -torch.sum(target_dist_projected * dist_action, dim=1)
        loss = (loss * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = loss.detach().abs().cpu() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        # Update target net
        if self.updates_done % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def _batch_obs(self, obs_list):
        # obs_list: list of (4,128,128) arrays
        # -> (batch,4,128,128)
        arrs = [self._preprocess_observation(o) for o in obs_list]
        return torch.cat(arrs, dim=0)