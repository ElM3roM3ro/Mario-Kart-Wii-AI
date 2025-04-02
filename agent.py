import math, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from collections import deque
import logging
import collections
from math import sqrt
from gym.wrappers import LazyFrames  # for memory-efficient frame storage

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
            return nn.functional.linear(
                input,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return nn.functional.linear(input, self.weight_mu, self.bias_mu)

# ----------------- Impala CNN Blocks ----------------- #
class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x + x_

class ImpalaCNNBlock(nn.Module):
    """
    A single IMPALA CNN block: one convolution followed by max-pooling and two residual blocks.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x

# ----------------- Updated BTRNetwork using new IMPALA setup ----------------- #
class BTRNetwork(nn.Module):
    def __init__(self, in_channels=4, num_actions=8, num_quantiles=8, n_cos=64,
                 model_size=4, hidden=256, spectral_norm='all'):
        super(BTRNetwork, self).__init__()
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.n_cos = n_cos

        # Define norm functions based on the spectral_norm parameter
        def identity(x): 
            return x
        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm in ['last', 'all']) else identity

        # Build the IMPALA CNN using the provided blocks
        self.cnn = nn.Sequential(
            ImpalaCNNBlock(in_channels, 16 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(16 * model_size, 32 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(32 * model_size, 32 * model_size, norm_func=norm_func_last),
            nn.ReLU()
        )
        # Adaptive maxpooling to 6x6 as per the guideline
        self.pool = nn.AdaptiveMaxPool2d((6, 6))
        # Compute feature dimension: 32 * model_size * 6 * 6
        self.feature_dim = 32 * model_size * 6 * 6

        # IQN: cosine embedding for tau samples
        self.cos_embedding = nn.Linear(n_cos, self.feature_dim)

        # Dueling streams using NoisyLinear layers
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_dim, hidden, std_init=0.5),
            nn.ReLU(),
            NoisyLinear(hidden, 1, std_init=0.5)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_dim, hidden, std_init=0.5),
            nn.ReLU(),
            NoisyLinear(hidden, num_actions, std_init=0.5)
        )

    def forward(self, x, taus=None):
        batch_size = x.size(0)
        features = self.cnn(x)
        features = self.pool(features)
        features = features.view(batch_size, -1)  # (B, feature_dim)
        if taus is None:
            taus = torch.rand(batch_size, self.num_quantiles, device=x.device)
        # IQN: compute cosine embeddings for tau samples
        i_pi = torch.arange(1, self.n_cos + 1, device=x.device).float() * math.pi
        taus_expanded = taus.unsqueeze(-1)  # (B, num_quantiles, 1)
        cos_emb = torch.cos(taus_expanded * i_pi)  # (B, num_quantiles, n_cos)
        phi = torch.relu(self.cos_embedding(cos_emb))  # (B, num_quantiles, feature_dim)
        # Modulate features with the IQN embeddings
        features = features.unsqueeze(1)  # (B, 1, feature_dim)
        modulated = features * phi         # (B, num_quantiles, feature_dim)
        modulated = modulated.view(batch_size * self.num_quantiles, self.feature_dim)
        # Dueling streams
        value = self.value_stream(modulated)       # (B*num_quantiles, 1)
        advantage = self.advantage_stream(modulated) # (B*num_quantiles, num_actions)
        value = value.view(batch_size, self.num_quantiles, 1)
        advantage = advantage.view(batch_size, self.num_quantiles, self.num_actions)
        # Combine streams ensuring that the advantage has zero mean
        q_values = value + advantage - advantage.mean(dim=2, keepdim=True)
        return q_values, taus

# ----------------- New Prioritized Replay Buffer using LazyFrames ----------------- #
class PrioritizedReplayBuffer:
    """
    Based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments.
    Uses LazyFrames for memory efficiency.
    """
    def __init__(self, burnin: int, capacity: int, gamma: float, n_step: int, parallel_envs: int, use_amp):
        self.burnin = burnin
        self.capacity = capacity  # ideally a power of two
        self.gamma = gamma
        self.n_step = n_step
        self.use_amp = use_amp
        self.n_step_buffers = [collections.deque(maxlen=self.n_step + 1) for _ in range(parallel_envs)]
        
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1.0
        
        self.data = [None for _ in range(self.capacity)]
        self.next_idx = 0
        self.size = 0

    @staticmethod
    def prepare_transition(state, next_state, action: int, reward: float, done: bool):
        # Convert action to a NumPy array first to avoid creating a tensor from a list of numpy.ndarrays.
        action = torch.tensor(np.array(action), device="cuda", dtype=torch.long)
        reward = torch.tensor(reward, device="cuda", dtype=torch.float)
        done = torch.tensor(done, device="cuda", dtype=torch.float)
        return state, next_state, action, reward, done

    def put(self, *transition, j):
        # transition format: (state, action, reward, next_state, done)
        self.n_step_buffers[j].append(transition)
        if len(self.n_step_buffers[j]) == self.n_step + 1 and not self.n_step_buffers[j][0][3]:
            state = self.n_step_buffers[j][0][0]
            action = self.n_step_buffers[j][0][1]
            next_state = self.n_step_buffers[j][self.n_step][0]
            done = self.n_step_buffers[j][self.n_step][3]
            reward = self.n_step_buffers[j][0][2]
            for k in range(1, self.n_step):
                reward += self.n_step_buffers[j][k][2] * self.gamma ** k
                if self.n_step_buffers[j][k][3]:
                    done = True
                    break

            assert isinstance(state, LazyFrames)
            assert isinstance(next_state, LazyFrames)

            idx = self.next_idx
            self.data[idx] = self.prepare_transition(state, next_state, action, reward, done)
            self.next_idx = (idx + 1) % self.capacity
            self.size = min(self.capacity, self.size + 1)
            prio = sqrt(self.max_priority)
            self._set_priority_min(idx, prio)
            self._set_priority_sum(idx, prio)

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[2 * idx] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    def sample(self, batch_size: int, beta: float) -> tuple:
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)
        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx
        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)
        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)
            weights[i] = weight / max_weight
        samples = [self.data[i] for i in indices]
        return indices, weights, self.prepare_samples(samples)

    def prepare_samples(self, batch):
        state, next_state, action, reward, done = zip(*batch)
        # Efficiently stack LazyFrames by converting each LazyFrame to a NumPy array and then stacking.
        state = torch.from_numpy(np.stack([s.__array__() for s in state]))
        next_state = torch.from_numpy(np.stack([s.__array__() for s in next_state]))
        action = torch.stack(action)
        reward = torch.stack(reward)
        done = torch.stack(done)
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    @property
    def burnedin(self):
        return self.size >= self.burnin

    def __len__(self):
        return self.size

# ----------------- BTR Agent incorporating IQN and Munchausen RL ----------------- #
class BTRAgent:
    def __init__(self,
                 state_shape=(4, 128, 128),
                 num_actions=8,
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
                 device='cuda',
                 parallel_envs=8):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.n_steps = n_steps
        self.munchausen_alpha = munchausen_alpha
        self.munchausen_tau = munchausen_tau
        self.munchausen_clip = munchausen_clip
        self.buffer = PrioritizedReplayBuffer(
            burnin=25000,
            capacity=buffer_size,
            gamma=self.gamma,
            n_step=self.n_steps,
            parallel_envs=parallel_envs,
            use_amp=False
        )
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
        self.online_net.eval()
        with torch.no_grad():
            obs = observation.unsqueeze(0).to(self.device)
            q_values, _ = self.online_net(obs)
            q_mean = q_values.mean(dim=1)
            action = q_mean.argmax(dim=1).item()
        return action

    def update(self, total_frames=None):
        if len(self.buffer) < 25000:
            if len(self.buffer) % 1000 == 0:
                logging.info(f"Buffer size: {len(self.buffer)}")
            return None

        self.online_net.train()
        self.target_net.train()
        indices, weights, (obs_batch, next_obs_batch, actions, rewards, dones) = self.buffer.sample(self.batch_size, beta=0.4)
        weights = torch.FloatTensor(weights).to(self.device)
        obs_batch = obs_batch.float().to(self.device)
        next_obs_batch = next_obs_batch.float().to(self.device)
        # Resize from raw size (e.g., 78x94) to (128, 128)
        #obs_batch = torch.nn.functional.interpolate(obs_batch, size=(128, 128), mode='bilinear', align_corners=False)
        #next_obs_batch = torch.nn.functional.interpolate(next_obs_batch, size=(128, 128), mode='bilinear', align_corners=False)
        actions = actions.long().to(self.device)
        rewards = rewards.float().to(self.device)
        dones = dones.float().to(self.device)

        # Compute current quantile estimates for taken actions
        quantiles, taus = self.online_net(obs_batch)
        actions_expanded = actions.unsqueeze(1).unsqueeze(2).expand(-1, self.online_net.num_quantiles, 1)
        current_quantiles = quantiles.gather(2, actions_expanded).squeeze(2)

        # Munchausen bonus computation
        q_values = quantiles.mean(dim=1)
        logits = q_values / self.munchausen_tau
        log_policy = torch.log_softmax(logits, dim=1)
        munchausen_bonus = torch.clamp(log_policy.gather(1, actions.unsqueeze(1)), min=self.munchausen_clip)
        rewards_aug = rewards + self.munchausen_alpha * munchausen_bonus.squeeze(1)

        # --- Soft target backup using softmax policy ---
        next_quantiles_target, _ = self.target_net(next_obs_batch)
        next_q_values_target = next_quantiles_target.mean(dim=1)  # shape: (batch, num_actions)
        logits_next = next_q_values_target / self.munchausen_tau
        pi_next = torch.softmax(logits_next, dim=1)  # soft policy over actions
        pi_next_expanded = pi_next.unsqueeze(1)  # shape: (batch, 1, num_actions)
        # Compute expected next quantile value as the weighted sum over actions
        expected_next_quantiles = (next_quantiles_target * pi_next_expanded).sum(dim=2)  # (batch, num_quantiles)
        target_quantiles = rewards_aug.unsqueeze(1) + self.gamma * expected_next_quantiles * (1 - dones.unsqueeze(1))
        target_quantiles = target_quantiles.detach()

        # Compute quantile regression loss between current and target quantiles
        taus = taus.unsqueeze(2)  # (batch, num_quantiles, 1)
        target_expanded = target_quantiles.unsqueeze(1)  # (batch, 1, num_quantiles)
        td_error = target_expanded - current_quantiles.unsqueeze(2)
        huber_loss = self._huber_loss(td_error, k=1.0)
        quantile_loss = torch.abs(taus - (td_error.detach() < 0).float()) * huber_loss
        sample_losses = quantile_loss.mean(dim=2).sum(dim=1)
        loss = (sample_losses * weights.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()

        new_priorities = sample_losses.detach().abs().cpu() + 1e-6
        self.buffer.update_priorities(indices, new_priorities)

        self.update_count += 1
        #logging.info(f"Update {self.update_count}: Loss = {loss.item():.4f}")
        if total_frames % 128000 == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            logging.info(f"Target network updated at update {self.update_count} with total frames {total_frames}")

        return loss.item()

    def _huber_loss(self, x, k=1.0):
        cond = (x.abs() <= k).float()
        loss = 0.5 * x.pow(2) * cond + k * (x.abs() - 0.5 * k) * (1 - cond)
        return loss
