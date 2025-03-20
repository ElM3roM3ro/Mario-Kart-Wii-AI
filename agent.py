import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------- Noisy Linear (with factorized noise) -------------- #
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.1):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.sigma_init / math.sqrt(self.in_features))
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.bias_sigma, self.sigma_init / math.sqrt(self.in_features))
    
    def reset_noise(self):
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        # outer product to create factorized noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)
    
    def _scale_noise(self, size, device):
        x = torch.randn(size, device=device)
        return x.sign() * torch.sqrt(x.abs() + 1e-10)

# -------------- Rainbow Network -------------- #
class RainbowNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=14, num_atoms=51, hidden=512):
        """
        This network is adapted from Kaixhin's implementation.
        Note: The conv architecture has been adjusted for 128x128 inputs.
        """
        super(RainbowNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        # Convolutional feature extractor (Kaixhin's default uses stride=1 for the third conv)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),   # 128 -> 31 (approx)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # 31 -> 14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # 14 -> 12
            nn.ReLU()
        )
        # Compute conv output size; for 128x128 this should be 64 * 12 * 12 = 9216
        test_in = torch.zeros(1, in_channels, 128, 128)
        conv_out_size = self.conv(test_in).view(1, -1).size(1)
        self.fc_input_dim = conv_out_size

        # Dueling streams using NoisyLinear layers
        self.value_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, hidden, sigma_init=0.1),
            nn.ReLU(),
            NoisyLinear(hidden, num_atoms, sigma_init=0.1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.fc_input_dim, hidden, sigma_init=0.1),
            nn.ReLU(),
            NoisyLinear(hidden, num_actions * num_atoms, sigma_init=0.1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        features = self.conv(x)
        features = features.view(batch_size, -1)
        value = self.value_stream(features)         # shape: (batch, num_atoms)
        advantage = self.advantage_stream(features)   # shape: (batch, num_actions*num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value.unsqueeze(1) + (advantage - advantage_mean)
        return q_dist
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# -------------- Prioritized Replay Buffer -------------- #
class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000000, alpha=0.6):
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
        weights = (total * total_prob) ** (-beta)
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
        num_actions=14,
        num_atoms=51,
        v_min=-1.0,
        v_max=1.0,
        gamma=0.99,
        lr=0.0001,
        buffer_size=1000000,
        batch_size=32,
        beta_start=0.4,
        beta_frames=100000,  # frames over which beta increases to 1.0
        target_update_interval=1000,
        n_steps=3,
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
        self.beta_increment_per_frame = (1.0 - beta_start) / beta_frames
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)

        self.online_net = RainbowNetwork(in_channels=state_shape[0],
                                         num_actions=num_actions,
                                         num_atoms=num_atoms,
                                         hidden=512).to(device)
        self.target_net = RainbowNetwork(in_channels=state_shape[0],
                                         num_actions=num_actions,
                                         num_atoms=num_atoms,
                                         hidden=512).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        self.frame_count = 0
        self.updates_done = 0
        self.target_update_interval = target_update_interval

        # Multi-step returns buffer (n_steps default set to 3 as in Rainbow)
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=self.n_steps)
    
    def select_action(self, observation):
        """
        Expects observation as a tensor of shape (4,128,128).
        Resets noise and selects an action based on the distributional Q-values.
        """
        self.online_net.reset_noise()
        self.online_net.eval()
        with torch.no_grad():
            x = observation.unsqueeze(0).to(self.device)
            dist = self.online_net(x)  # (1, num_actions, num_atoms)
            dist = torch.softmax(dist, dim=2)
            q_values = torch.sum(dist * self.support.view(1, 1, -1), dim=2)
            action = q_values.argmax(dim=1).item()
        return action
    
    def store_transition(self, transition):
        """
        transition: (obs, action, reward, next_obs, done)
        Uses an n-step buffer to compute multi-step returns.
        """
        self.n_step_buffer.append(transition)
        # If terminal, flush the n-step buffer.
        if transition[4]:
            self._flush_n_step_buffer()
        # Otherwise, if we have enough steps, compute multi-step return.
        elif len(self.n_step_buffer) >= self.n_steps:
            R = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(self.n_steps)])
            next_obs = self.n_step_buffer[-1][3]
            done = any(t[4] for t in self.n_step_buffer)
            obs, action, _, _, _ = self.n_step_buffer[0]
            self.buffer.store((obs, action, R, next_obs, done))
            self.n_step_buffer.popleft()
    
    def _flush_n_step_buffer(self):
        while self.n_step_buffer:
            n = len(self.n_step_buffer)
            R = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(n)])
            next_obs = self.n_step_buffer[-1][3]
            done = any(t[4] for t in self.n_step_buffer)
            obs, action, _, _, _ = self.n_step_buffer[0]
            self.buffer.store((obs, action, R, next_obs, done))
            self.n_step_buffer.popleft()
    
    def update(self):
        if len(self.buffer.buffer) < self.batch_size:
            return

        self.online_net.reset_noise()
        self.target_net.reset_noise()

        self.updates_done += 1
        self.frame_count += 1
        beta = min(1.0, self.beta_start + self.beta_increment_per_frame * self.frame_count)
        samples, indices, weights = self.buffer.sample(self.batch_size, beta=beta)
        weights = weights.to(self.device)

        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
        obs_batch_t = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in obs_batch]).to(self.device)
        next_obs_batch_t = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in next_obs_batch]).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Current distribution from online network.
        dist = self.online_net(obs_batch_t)  # (batch, num_actions, num_atoms)
        log_p = torch.log_softmax(dist, dim=2)
        log_p_a = log_p.gather(1, actions_t.unsqueeze(2).expand(self.batch_size, 1, self.num_atoms)).squeeze(1)

        with torch.no_grad():
            # Use online net to select best action, target net for evaluation.
            next_dist = self.online_net(next_obs_batch_t)
            next_dist = torch.softmax(next_dist, dim=2)
            q_next = torch.sum(next_dist * self.support.view(1, 1, -1), dim=2)
            best_actions = q_next.argmax(dim=1)
            target_dist = self.target_net(next_obs_batch_t)
            target_dist = torch.softmax(target_dist, dim=2)
            target_dist = target_dist[range(self.batch_size), best_actions]

        Tz = rewards_t + (1 - dones_t) * self.gamma * self.support.view(1, -1)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).to(self.device)
        target_proj = torch.zeros_like(target_dist)
        target_proj.view(-1).index_add_(0, (l + offset).view(-1),
                                         (target_dist * (u.float() - b)).view(-1))
        target_proj.view(-1).index_add_(0, (u + offset).view(-1),
                                         (target_dist * (b - l.float())).view(-1))

        loss = -(target_proj * log_p_a).sum(1)
        loss = (loss * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = loss.detach().abs().cpu() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        if self.updates_done % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        if self.updates_done % 100 == 0:
            print(f"Update {self.updates_done}: Loss = {loss.item():.4f}")
