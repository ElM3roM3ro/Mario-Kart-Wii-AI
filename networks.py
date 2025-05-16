"""
This file defines all the neural network architectures available to use.
"""
from functools import partial
from math import sqrt
import math

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import time
#from torchvision.utils import save_image 

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

  @torch.no_grad()
  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  @torch.no_grad()
  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  @torch.no_grad()
  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0=0.5, self_norm=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()

        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in

        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    #@torch.autocast('cuda')
    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))

class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func, activation=nn.ReLU):
        super().__init__()

        self.activation = activation()

        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    #@torch.autocast('cuda')
    def forward(self, x):
        #if x.abs().sum().item() != 0:
        x_ = self.conv_0(self.activation(x))
        #if x_.abs().sum().item() == 0:
        #raise Exception("0 tensor found within residual layer!")
        x_ = self.conv_1(self.activation(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func, activation=nn.ReLU, layer_norm=False,
                 layer_norm_shapes=False):
        super().__init__()
        self.layer_norm = layer_norm

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)

        if self.layer_norm:
            self.norm_layer1 = nn.LayerNorm(layer_norm_shapes[0])
            #self.norm_layer2 = nn.LayerNorm(layer_norm_shapes[1])

        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)

    #@torch.autocast('cuda')
    def forward(self, x):
        x = self.conv(x)

        if self.layer_norm:
            x = self.norm_layer1(x)

        #raise Exception("Array of 0s!")
        #print(x.abs().sum().item())
        x = self.max_pool(x)

        x = self.residual_0(x)

        x = self.residual_1(x)

        #if self.layer_norm:
        #x = self.norm_layer2(x)

        return x

class ImpalaCNNLargeIQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, model_size=2, spectral=True, device='cuda:0',
                 noisy=False, maxpool=False, num_tau=8, maxpool_size=6, dueling=True,
                 linear_size=512, ncos=64, arch="impala", layer_norm=False,
                 activation="relu"):
        super().__init__()

        self.start = time.time()
        self.model_size = model_size
        self.actions = actions
        self.device = device
        self.noisy = noisy
        self.maxpool = maxpool
        self.dueling = dueling
        self.in_depth = in_depth

        self.activation = activation
        conv_activation = nn.ReLU


        self.linear_size = linear_size
        self.num_tau = num_tau

        self.maxpool_size = maxpool_size

        self.layer_norm = layer_norm

        self.n_cos = ncos
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)
        self.arch = arch

        if noisy:
            linear_layer = FactorizedNoisyLinear
        else:
            linear_layer = nn.Linear

        def identity(p): return p

        if spectral:
            norm_func = torch.nn.utils.parametrizations.spectral_norm
        else:
            norm_func = identity


        self.conv = nn.Sequential(
              ImpalaCNNBlock(in_depth, int(16*model_size), norm_func=norm_func, activation=conv_activation,
                             layer_norm=self.layer_norm,
                             layer_norm_shapes=([int(16*model_size), 84, 84], [int(16*model_size), 42, 42])),
              ImpalaCNNBlock(int(16*model_size), int(32*model_size), norm_func=norm_func, activation=conv_activation,
                              layer_norm=self.layer_norm,
                             layer_norm_shapes=([int(32*model_size), 42, 42], [int(32*model_size), 21, 21])),
              ImpalaCNNBlock(int(32*model_size), int(32*model_size), norm_func=norm_func, activation=conv_activation,
                              layer_norm=self.layer_norm,
                             layer_norm_shapes=([int(32*model_size), 21, 21], [int(32*model_size), 11, 11])),
              nn.ReLU()
          )

        if self.maxpool:
              self.pool = torch.nn.AdaptiveMaxPool2d((self.maxpool_size, self.maxpool_size))
              if self.maxpool_size == 8:
                  self.conv_out_size = 2048 * model_size
              elif self.maxpool_size == 6:
                  self.conv_out_size = int(1152 * model_size)
              elif self.maxpool_size == 4:
                  self.conv_out_size = 512 * model_size
              else:
                  raise Exception("No Conv out size for this maxpool size")
        else:
              self.conv_out_size = int(32 * model_size * 11 * 11)

        self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)

        if self.dueling:
            if not self.layer_norm:
                self.dueling = Dueling(
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  conv_activation(),
                                  linear_layer(self.linear_size, 1)),
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  conv_activation(),
                                  linear_layer(self.linear_size, actions))
                )
            else:
                # torch.nn.utils.parametrizations.spectral_norm

                self.dueling = Dueling(
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  nn.LayerNorm(self.linear_size),
                                  conv_activation(),
                                  linear_layer(self.linear_size, 1)),
                    nn.Sequential(linear_layer(self.conv_out_size, self.linear_size),
                                  nn.LayerNorm(self.linear_size),
                                  conv_activation(),
                                  linear_layer(self.linear_size, actions))
                )
        else:
            self.linear_layers = nn.Sequential(
                    linear_layer(self.conv_out_size, self.linear_size),
                    conv_activation(),
                    linear_layer(self.linear_size, actions))

        self.to(device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    #@torch.autocast('cuda')
    def forward(self, inputt, advantages_only=False):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        batch_size = inputt.size()[0]
        if self.arch == "each_frame":
            inputt = inputt.reshape((-1, 1, 84, 84))

        #print("Forward Func")
        inputt = inputt.float() / 255
        #print(input.abs().sum().item())

        x = self.conv(inputt)
        #print(x.device)
        if self.maxpool and (self.arch == "impala" or self.arch == "each_frame" or self.arch == "3d"):
            x = self.pool(x)

        #print(x.device)
        x = x.view(batch_size, -1)

        cos, taus = self.calc_cos(batch_size, self.num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.num_tau, self.conv_out_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.conv_out_size)

        if self.dueling:
            out = self.dueling(x, advantages_only=advantages_only)
        else:
            out = self.linear_layers(x)

        #print(out.device)
        return out.view(batch_size, self.num_tau, self.actions), taus

    def qvals(self, inputs, advantages_only=False):
        quantiles, _ = self.forward(inputs, advantages_only)

        actions = quantiles.mean(dim=1)

        return actions

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).to(self.device).unsqueeze(-1) #(batch_size, n_tau, 1)
        cos = torch.cos(taus*self.pis)

        #assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self, name):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load(name))