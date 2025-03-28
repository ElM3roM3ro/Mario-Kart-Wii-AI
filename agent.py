import math, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import logging

# ----------------- NoisyLinear (with σ = 0.5) ----------------- #
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return nn.functional.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return nn.functional.linear(input, self.weight_mu, self.bias_mu)

# ----------------- BTR Network with IQN and Impala Blocks ----------------- #
class BTRNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=14, num_quantiles=8, n_cos=64, hidden=512):
        super(BTRNetwork, self).__init__()
        width_scale = 2  # Impala width scale factor

        # Impala-style feature extractor with spectral normalization and adaptive max pooling:
        self.conv1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 32 * width_scale, kernel_size=8, stride=4)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(32 * width_scale, 64 * width_scale, kernel_size=4, stride=2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64 * width_scale, 64 * width_scale, kernel_size=3, stride=1)),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool2d((6, 6))  # Fixed output: 6x6 feature map
        self.feature_dim = (64 * width_scale) * 6 * 6  # 128 * 6 * 6 = 4608

        # IQN: cosine embedding for tau samples
        self.n_cos = n_cos
        self.cos_embedding = nn.Linear(n_cos, self.feature_dim)

        # Dueling streams with NoisyLinear layers (using σ = 0.5)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_dim, hidden, sigma_init=0.5),
            nn.ReLU(),
            NoisyLinear(hidden, 1, sigma_init=0.5)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_dim, hidden, sigma_init=0.5),
            nn.ReLU(),
            NoisyLinear(hidden, num_actions, sigma_init=0.5)
        )
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles

    def forward(self, x, taus=None):
        batch_size = x.size(0)
        # Feature extraction
        features = self.conv1(x)
        features = self.conv2(features)
        features = self.conv3(features)
        features = self.pool(features)
        features = features.view(batch_size, -1)  # shape: (batch, feature_dim)

        # Sample taus if not provided
        if taus is None:
            taus = torch.rand(batch_size, self.num_quantiles, device=x.device)  # shape: (batch, num_quantiles)

        # Cosine embedding: compute phi(τ) = ReLU(Linear(cos(π * i * τ))) for i=1,…,n_cos
        i_pi = torch.arange(1, self.n_cos + 1, device=x.device).float() * math.pi
        taus_expanded = taus.unsqueeze(-1)  # (batch, num_quantiles, 1)
        cos_emb = torch.cos(taus_expanded * i_pi)  # (batch, num_quantiles, n_cos)
        phi = torch.relu(self.cos_embedding(cos_emb))  # (batch, num_quantiles, feature_dim)

        # Modulate features with the tau embedding
        features = features.unsqueeze(1)  # (batch, 1, feature_dim)
        modulated = features * phi         # (batch, num_quantiles, feature_dim)
        modulated = modulated.view(batch_size * self.num_quantiles, self.feature_dim)

        # Compute dueling streams
        value = self.value_stream(modulated)         # (batch*num_quantiles, 1)
        advantage = self.advantage_stream(modulated)   # (batch*num_quantiles, num_actions)
        value = value.view(batch_size, self.num_quantiles, 1)
        advantage = advantage.view(batch_size, self.num_quantiles, self.num_actions)
        # Combine streams (dueling architecture)
        q_values = value + advantage - advantage.mean(dim=2, keepdim=True)  # (batch, num_quantiles, num_actions)
        return q_values, taus

# ----------------- Prioritized Replay Buffer (unchanged except PER α) ----------------- #
class PrioritizedReplayBuffer:
    def __init__(self, capacity=1048576, alpha=0.2):  # capacity updated to 1,048,576
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

# ----------------- BTR Agent incorporating IQN and Munchausen RL ----------------- #
class BTRAgent:
    def __init__(self,
                 state_shape=(4, 128, 128),
                 num_actions=14,
                 num_quantiles=8,
                 gamma=0.997,
                 lr=1e-4,
                 buffer_size=1048576,
                 batch_size=256,
                 target_update_interval=500,
                 n_steps=3,
                 munchausen_alpha=0.9,
                 munchausen_tau=0.03,
                 munchausen_clip=-1.0,
                 device='cuda'):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.n_steps = n_steps
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_tau = munchausen_tau
        self.munchausen_clip = munchausen_clip
        self.buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.2)
        self.n_step_buffer = deque(maxlen=n_steps)
        self.online_net = BTRNetwork(in_channels=state_shape[0],
                                     num_actions=num_actions,
                                     num_quantiles=num_quantiles).to(device)
        self.target_net = BTRNetwork(in_channels=state_shape[0],
                                     num_actions=num_actions,
                                     num_quantiles=num_quantiles).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=1.95e-5, betas=(0.9, 0.999))
        self.target_update_interval = target_update_interval
        self.update_count = 0

    def select_action(self, observation):
        # Expects observation as tensor of shape (4,128,128)
        self.online_net.eval()
        with torch.no_grad():
            obs = observation.unsqueeze(0).to(self.device)
            q_values, _ = self.online_net(obs)  # (1, num_quantiles, num_actions)
            q_mean = q_values.mean(dim=1)  # average over quantile samples: (1, num_actions)
            action = q_mean.argmax(dim=1).item()
        return action

    def store_transition(self, transition):
        # transition: (obs, action, reward, next_obs, done)
        self.n_step_buffer.append(transition)
        if transition[4]:
            self._flush_n_step_buffer()
        elif len(self.n_step_buffer) >= self.n_steps:
            R = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(self.n_steps)])
            next_obs = self.n_step_buffer[-1][3]
            done = any(t[4] for t in self.n_step_buffer)
            obs, action, _, _, _ = self.n_step_buffer[0]
            self.buffer.store((obs, action, R, next_obs, done), priority=1.0)
            self.n_step_buffer.popleft()

    def _flush_n_step_buffer(self):
        while self.n_step_buffer:
            n = len(self.n_step_buffer)
            R = sum([self.gamma**i * self.n_step_buffer[i][2] for i in range(n)])
            next_obs = self.n_step_buffer[-1][3]
            done = any(t[4] for t in self.n_step_buffer)
            obs, action, _, _, _ = self.n_step_buffer[0]
            self.buffer.store((obs, action, R, next_obs, done), priority=1.0)
            self.n_step_buffer.popleft()

    def update(self, total_frames=None):
        # Only update if a minimum of 200K transitions are in the replay buffer.
        if len(self.buffer.buffer) < 25000:
            if len(self.buffer.buffer) % 1000 == 0:
                logging.info(f"{len(self.buffer.buffer)}")
            return None

        self.online_net.train()
        self.target_net.train()
        samples, indices, weights = self.buffer.sample(self.batch_size, beta=0.4)
        weights = weights.to(self.device)
        obs_batch, actions, rewards, next_obs_batch, dones = zip(*samples)
        obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in obs_batch]).to(self.device)
        next_obs_batch = torch.cat([torch.from_numpy(o).unsqueeze(0).float() for o in next_obs_batch]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # --- Current state quantile estimates ---
        quantiles, taus = self.online_net(obs_batch)  # (batch, num_quantiles, num_actions)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, self.online_net.num_quantiles, 1)
        current_quantiles = quantiles.gather(2, actions_expanded).squeeze(2)  # (batch, num_quantiles)

        # --- Munchausen bonus ---
        q_values = quantiles.mean(dim=1)  # (batch, num_actions)
        logits = q_values / self.munchausen_tau
        log_policy = torch.log_softmax(logits, dim=1)
        munchausen_bonus = torch.clamp(log_policy.gather(1, actions.unsqueeze(1)), min=self.munchausen_clip)
        rewards_aug = rewards + self.munchausen_alpha * munchausen_bonus.squeeze(1)

        # --- Next state quantile estimation ---
        next_quantiles_online, _ = self.online_net(next_obs_batch)
        next_q_values_online = next_quantiles_online.mean(dim=1)
        best_actions = next_q_values_online.argmax(dim=1)
        best_actions_expanded = best_actions.unsqueeze(1).unsqueeze(2).expand(-1, self.online_net.num_quantiles, 1)
        next_quantiles_target, _ = self.target_net(next_obs_batch)
        next_quantiles = next_quantiles_target.gather(2, best_actions_expanded).squeeze(2)  # (batch, num_quantiles)

        # --- Compute target quantiles ---
        target_quantiles = rewards_aug.unsqueeze(1) + self.gamma * next_quantiles * (1 - dones.unsqueeze(1))
        target_quantiles = target_quantiles.detach()

        # --- Quantile Huber loss ---
        taus = taus.unsqueeze(2)  # (batch, num_quantiles, 1)
        target_expanded = target_quantiles.unsqueeze(1)  # (batch, 1, num_quantiles)
        td_error = target_expanded - current_quantiles.unsqueeze(2)  # (batch, num_quantiles, num_quantiles)
        huber_loss = self._huber_loss(td_error, k=1.0)
        quantile_loss = (torch.abs(taus - (td_error.detach() < 0).float()) * huber_loss) / 1.0
        sample_losses = quantile_loss.mean(dim=2).sum(dim=1)
        loss = (sample_losses * weights.squeeze()).mean()

        # --- Gradient step with clipping ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        # --- Update priorities ---
        new_priorities = sample_losses.detach().abs().cpu() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        self.update_count += 1

        # --- Logging ---
        logging.info(f"Update {self.update_count}: Loss = {loss.item():.4f}")
        if self.update_count % self.target_update_interval == 0:
            if total_frames is not None:
                logging.info(f"Target network updated at update {self.update_count} with total frames {total_frames}")
            else:
                logging.info(f"Target network updated at update {self.update_count}")
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def _huber_loss(self, x, k=1.0):
        cond = (x.abs() <= k).float()
        loss = 0.5 * x.pow(2) * cond + k * (x.abs() - 0.5 * k) * (1 - cond)
        return loss
