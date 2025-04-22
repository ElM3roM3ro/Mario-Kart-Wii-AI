import os
import numpy as np
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from functools import partial
from networks import ImpalaCNNLargeIQN, FactorizedNoisyLinear
from PER_btr import PER
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("agent_out.log", mode='w'),
        logging.StreamHandler()
    ]
)

class EpsilonGreedy:
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)

def randomise_action_batch(x, probs, n_actions):
    mask = torch.rand(x.shape) < probs
    random_values = torch.randint(0, n_actions, x.shape)
    x[mask] = random_values[mask]
    return x

def choose_eval_action(observation, eval_net, n_actions, device, rng):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()
        if rng > 0.:
            x = randomise_action_batch(x, 0.01, n_actions)
    return x

def create_network(impala, iqn, input_dims, n_actions, spectral_norm, device, noisy, maxpool, model_size, maxpool_size,
                   linear_size, num_tau, dueling, ncos, non_factorised, arch,
                   layer_norm=False, activation="relu", c51=False):
    if impala:
        if iqn:
            return ImpalaCNNLargeIQN(input_dims[0], n_actions, spectral=spectral_norm, device=device, noisy=noisy,
                                     maxpool=maxpool, model_size=model_size, num_tau=num_tau, maxpool_size=maxpool_size,
                                     dueling=dueling, linear_size=linear_size, ncos=ncos,
                                     arch=arch, layer_norm=layer_norm, activation=activation)
        # if c51:
        #     return ImpalaCNNLargeC51(input_dims[0], n_actions, spectral=spectral_norm, device=device,
        #                             noisy=noisy, maxpool=maxpool, model_size=model_size, linear_size=linear_size)
        # else:
        #     return ImpalaCNNLarge(input_dims[0], n_actions, spectral=spectral_norm, device=device,
        #                           noisy=noisy, maxpool=maxpool, model_size=model_size, maxpool_size=maxpool_size,
        #                           linear_size=linear_size)
    else:
        print("ERROR: Model doesn't exist")

# Helper function to convert an object to a NumPy array only if necessary.
def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, 'cpu'):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)

class BTRAgent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=256,
                 rr=1, maxpool_size=6, lr=1e-4, target_replace=500,
                 noisy=True, spectral=True, munch=True, iqn=True, double=False, dueling=True, impala=True,
                 discount=0.997, per=True,
                 taus=8, model_size=2, linear_size=512, ncos=64, rainbow=False, maxpool=True,
                 non_factorised=False, replay_period=1, analytics=False, framestack=4,
                 rgb=False, imagex=128, imagey=128, arch='impala', per_alpha=0.2,
                 per_beta_anneal=False, layer_norm=False, max_mem_size=1048576, c51=False,
                 eps_steps=2000000, eps_disable=True,
                 activation="relu", n=3, munch_alpha=0.9,
                 grad_clip=10):
        # Set up parameters (defaults taken from Agent_btr.py, with image dimensions updated)
        self.per_alpha = per_alpha if not rainbow else 0.5
        self.procgen = True if input_dims[1] == 64 else False
        self.grad_clip = grad_clip
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing
        self.activation = activation
        self.layer_norm = layer_norm
        self.loading_checkpoint = False
        self.per_beta = 0.45
        self.per_beta_anneal = per_beta_anneal
        if self.per_beta_anneal:
            self.per_beta = 0
        self.replay_ratio = int(rr) if rr > 0.99 else float(rr)
        self.total_frames = total_frames
        self.num_envs = num_envs
        self.min_sampling_size = 4000 if testing else 200000
        self.lr = lr
        self.analytics = analytics
        # if self.analytics:
        #     from Analytic import Analytics
        #     self.analytic_object = Analytics(agent_name, testing)
        self.replay_period = replay_period
        self.total_grad_steps = (self.total_frames - self.min_sampling_size) / (self.replay_period / self.replay_ratio)
        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0
        self.n = n
        self.gamma = discount
        self.batch_size = batch_size
        self.model_size = model_size
        self.maxpool_size = maxpool_size
        self.spectral_norm = spectral
        self.noisy = noisy
        self.non_factorised = non_factorised
        self.impala = impala
        self.dueling = dueling
        self.c51 = c51
        self.iqn = iqn
        self.ncos = ncos
        self.double = double
        self.maxpool = maxpool
        self.munchausen = munch
        if self.munchausen:
            self.entropy_tau = 0.03
            self.lo = -1
            self.alpha = munch_alpha
        self.max_mem_size = max_mem_size
        self.replace_target_cnt = target_replace
        self.loss_type = "huber"
        if self.loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.num_tau = taus
        if self.loading_checkpoint:
            self.min_sampling_size = 300000
        self.Vmax = 10
        self.Vmin = -10
        self.N_ATOMS = 51
        if not self.loading_checkpoint and not self.testing:
            self.eps_start = 1.0
            self.eps_steps = eps_steps
            self.eps_final = 0.01
        else:
            self.eps_start = 0.00
            self.eps_steps = eps_steps
            self.eps_final = 0.00
        self.eps_disable = eps_disable
        self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)
        self.per = per
        self.linear_size = linear_size
        self.arch = arch
        self.framestack = framestack
        self.rgb = rgb
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=self.rgb, imagex=imagex, imagey=imagey)
        self.network_creator_fn = partial(create_network, self.impala, self.iqn, self.input_dims, self.n_actions,
                                          self.spectral_norm, self.device,
                                          self.noisy, self.maxpool, self.model_size, self.maxpool_size,
                                          self.linear_size,
                                          self.num_tau, self.dueling, self.ncos,
                                          self.non_factorised, self.arch, layer_norm=self.layer_norm,
                                          activation=self.activation, c51=self.c51)
        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)
        self.net.train()
        self.eval_net = None
        for param in self.tgt_net.parameters():
            param.requires_grad = False
        self.env_steps = 0
        self.grad_steps = 0
        self.replay_ratio_cnt = 0
        self.eval_mode = False
        if self.loading_checkpoint:
            self.load_models(self.agent_name)
        self.scaler = torch.amp.GradScaler(device=device)
    
    def get_grad_steps(self):
        return self.grad_steps

    def prep_evaluation(self):
        self.eval_net = deepcopy(self.net)
        self.disable_noise(self.eval_net)

    @torch.no_grad()
    def reset_noise(self, net):
        import networks
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        import networks
        for m in net.modules():
            if isinstance(m, networks.FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation, debug=False):
        with T.no_grad():
            # For noisy networks make sure to reset noise if in training mode.
            if self.noisy and not self.eval_mode:
                self.reset_noise(self.net)
                
            # Convert observation into a tensor and compute Q-values.
            state = T.tensor(observation, dtype=T.float).to(self.device)
            qvals = self.net.qvals(state, advantages_only=True)

            # --- Debug Code Start ---
            if debug:
                # Log the entire Q-values tensor.
                logging.info("DEBUG: Q-values from network: %s", qvals.cpu().numpy())
                # Compute the best (argmax) action from the Q-values.
                best_action = T.argmax(qvals, dim=1)
                logging.info("DEBUG: Action chosen based on max Q: %s", best_action.cpu().numpy())
            # --- Debug Code End ---
            
            # Choose action using the greedy action from the network output.
            x = T.argmax(qvals, dim=1).cpu()
            
            # If in early training (or other conditions), add some randomness.
            if self.env_steps < self.min_sampling_size or not self.noisy or (self.env_steps < self.total_frames / 2 and self.eps_disable):
                probs = self.epsilon.eps
                x = randomise_action_batch(x, probs, self.n_actions)
            return x

    # Updated store_transition that avoids unnecessary conversions.
    def store_transition(self, state, action, reward, next_state, done, stream, prio=True):
        if not isinstance(state, np.ndarray):
            state = to_numpy(state)
        if not isinstance(next_state, np.ndarray):
            next_state = to_numpy(next_state)
        self.memory.append(state, action, reward, next_state, done, stream, prio=prio)
        self.epsilon.update_eps()
        self.env_steps += 1

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        print("Saving Models")
        self.net.save_checkpoint(self.agent_name)
        print("Models Saved")

    def load_models(self, name):
        print("Loading Models")
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)
        print("Models Loaded")

    def learn(self):
        if self.replay_ratio < 1:
            if self.replay_ratio_cnt == 0:
                self.learn_call()
            self.replay_ratio_cnt = (self.replay_ratio_cnt + 1) % int(1 / self.replay_ratio)
        else:
            for i in range(self.replay_ratio):
                self.learn_call()

    def learn_call(self):
        if self.env_steps < self.min_sampling_size:
            return

        if self.per and self.per_beta_anneal:
            self.memory.beta = min(self.memory.beta + self.priority_weight_increase, 1)

        if self.noisy:
            self.reset_noise(self.tgt_net)

        if self.grad_steps % self.replace_target_cnt == 0:
            print("Updating Target Network")
            self.replace_target_network()

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)
        # # --- Debug Code Start ---
        # with T.no_grad():
        #     # Compute Q-values for the states in the current training batch.
        #     batch_qvals = self.net.qvals(states, advantages_only=True)
        #     # Compute best actions from these Q-values.
        #     batch_best_actions = T.argmax(batch_qvals, dim=1)
        #     logging.info("DEBUG: Training Batch Q-values:\n%s", batch_qvals.cpu().numpy())
        #     logging.info("DEBUG: Training Batch - Chosen actions (by max Q):\n%s", batch_best_actions.cpu().numpy())
        # # --- Debug Code End ---

        self.optimizer.zero_grad()

        #use this code to check your states are correct
        # x = np.random.randint(0,200)
        # print(dones[x])
        # print(rewards[x])
        # fig, axes = plt.subplots(1, 4, figsize=(12, 3))

        # axes[0].imshow(states[x][0].unsqueeze(0).cpu().permute(1, 2, 0))
        # axes[0].set_title("state first frame")
        # axes[0].axis("off")

        # axes[1].imshow(next_states[x][0].unsqueeze(0).cpu().permute(1, 2, 0))
        # axes[1].set_title("next state first frame")
        # axes[1].axis("off")

        # axes[2].imshow(states[x][-1].unsqueeze(0).cpu().permute(1, 2, 0))
        # axes[2].set_title("state last frame")
        # axes[2].axis("off")

        # axes[3].imshow(next_states[x][-1].unsqueeze(0).cpu().permute(1, 2, 0))
        # axes[3].set_title("next state last frame")
        # axes[3].axis("off")

        # plt.tight_layout()
        # plt.show()
        
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()

        with torch.amp.autocast(device_type=self.device):
            if self.iqn and self.munchausen:
                with torch.no_grad():
                    Q_targets_next, _ = self.tgt_net(next_states)
                    q_t_n = Q_targets_next.mean(dim=1)
                    actions = actions.unsqueeze(1)
                    rewards = rewards.unsqueeze(1)
                    dones = dones.unsqueeze(1)
                    if self.per:
                        weights = weights.unsqueeze(1)
                    logsum = torch.logsumexp((q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)
                    tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)
                    pi_target = T.nn.functional.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)
                    Q_target = (self.gamma ** self.n * (pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
                    q_k_target = self.net.qvals(states)
                    v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
                    tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp((q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)
                    munchausen_addon = tau_log_pik.gather(1, actions)
                    munchausen_reward = (rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
                    Q_targets = munchausen_reward + Q_target
                q_k, taus = self.net(states)
                Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))
                td_error = Q_targets - Q_expected
                loss_v = T.abs(td_error).sum(dim=1).mean(dim=1).data
                huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)
                quantil_l = T.abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0
                loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)
                if self.per:
                    loss = loss * weights.to(self.net.device)
                loss = loss.mean()

        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())

        self.scaler.scale(loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        T.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")

def calculate_huber_loss(td_errors, k=1.0, taus=8):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], taus, taus), "huber loss has wrong shape"
    return loss