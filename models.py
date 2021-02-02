import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: using multiplicative Gaussian noise here
class CoupledGaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(CoupledGaussianDropout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Sample noise   e ~ N(0, alpha)
        Add noise h = h_ * (1 + e)
        """
        epsilon = torch.randn_like(x) * self.alpha + 1

        return x * epsilon


class STPNet(nn.Module):
    def __init__(self,
                 input_dim=64,
                 hidden_dim=16,
                 noise_std=0.0,
                 syn_tau=6,      # syn_tau: recovery time constant
                 syn_u=0.5):     # syn_u: calcium concentration

        super(STPNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.syn_tau = syn_tau
        self.syn_u = syn_u

        self.noise = CoupledGaussianDropout(
            alpha=noise_std) if noise_std > 0 else None
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def init_syn_x(self, batch_size=128):
        """Initialize syn_x for the input units."""
        return torch.ones([batch_size, self.input_dim])

    def forward(self, inputs):
        # add noise
        if self.noise:
            inputs = F.relu(self.noise(inputs))

        k = (1 / self.syn_tau) + self.syn_u * inputs
        syn_x_list = [self.syn_x]
        for i in range(inputs.shape[1]-1):
            # update synaptic plasticity
            # backward Euler
            self.syn_x = (1 / k[:, i]) * ((1 / self.syn_tau) -
                                          ((1 / self.syn_tau) -
                                           self.syn_x * k[:, i]) *
                                          torch.exp(-k[:, i]))
            # # forward Euler
            # self.syn_x = self.syn_x + (1 - self.syn_x) / self.syn_tau - \
            #     self.syn_u * self.syn_x * inputs[:, i]
            # # clamp between [0,1]
            # self.syn_x = torch.clamp(self.syn_x, min=0, max=1)

            syn_x_list.append(self.syn_x)

        input_syn = torch.stack(syn_x_list, dim=1)
        hidden = F.relu(self.linear1(input_syn * inputs))
        if self.noise:
            hidden = F.relu(self.noise(hidden))
        output = self.linear2(hidden)

        return output, hidden, inputs, input_syn


# Adapted from: https://mlexplained.com/2019/02/15/building-an-lstm-from-scratch-in-pytorch-lstms-in-depth-part-1/
class OptimizedRNN(nn.Module):
    def __init__(self,
                 input_dim=64,
                 hidden_dim=16,
                 noise_std=0.0):

        super(OptimizedRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.noise = CoupledGaussianDropout(
            alpha=noise_std) if noise_std > 0 else None

        self.register_parameter(
            'weight_ih', nn.Parameter(torch.Tensor(input_dim, hidden_dim)))
        self.register_parameter(
            'weight_hh', nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)))
        self.register_parameter(
            'bias', nn.Parameter(torch.Tensor(hidden_dim)))
        self.init_weights()

        self.linear = nn.Linear(hidden_dim, 1)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def init_hidden(self, batch_size=128):
        """Initialize hidden state of RNN."""
        return torch.zeros([1, batch_size, self.hidden_dim])

    def forward(self, inputs):
        """Assumes input is of shape (batch, sequence, feature)"""
        # add noise
        if self.noise:
            inputs = F.relu(self.noise(inputs))

        hidden = []
        for i in range(inputs.shape[1]):
            x = inputs[:, i, :]
            # batch the computations into a single matrix multiplication
            self.hidden = x @ self.weight_ih + self.hidden @ self.weight_hh + self.bias
            self.hidden = F.relu(self.hidden)
            if self.noise:
                self.hidden = F.relu(self.noise(self.hidden))

            hidden.append(self.hidden)

        hidden = torch.cat(hidden, dim=0)
        hidden = hidden.transpose(0, 1).contiguous()
        output = self.linear(hidden)

        return output, hidden, inputs


class STPRNN(nn.Module):
    def __init__(self,
                 input_dim=64,
                 hidden_dim=16,
                 noise_std=0.0,
                 syn_tau=6,      # syn_tau: recovery time constant
                 syn_u=0.5):     # syn_u: calcium concentration

        super(STPRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.syn_tau = syn_tau
        self.syn_u = syn_u

        self.noise = CoupledGaussianDropout(
            alpha=noise_std) if noise_std > 0 else None

        self.register_parameter(
            'weight_ih', nn.Parameter(torch.Tensor(input_dim * 2, hidden_dim)))
        self.register_parameter(
            'weight_hh', nn.Parameter(torch.Tensor(hidden_dim, hidden_dim)))
        self.register_parameter(
            'bias', nn.Parameter(torch.Tensor(hidden_dim)))
        self.init_weights()

        self.linear = nn.Linear(hidden_dim, 1)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def init_syn_x(self, batch_size=128):
        """Initialize syn_x for the input units."""
        return torch.ones([batch_size, self.input_dim])

    def init_hidden(self, batch_size=128):
        """Initialize hidden state of RNN."""
        return torch.zeros([1, batch_size, self.hidden_dim])

    def forward(self, inputs):
        """Assumes input is of shape (batch, sequence, feature)"""
        # add noise
        if self.noise:
            inputs = F.relu(self.noise(inputs))

        k = (1 / self.syn_tau) + self.syn_u * inputs
        syn_x_list = [self.syn_x]
        for i in range(inputs.shape[1]-1):
            # update synaptic plasticity
            # backward Euler
            self.syn_x = (1 / k[:, i]) * ((1 / self.syn_tau) -
                                          ((1 / self.syn_tau) -
                                           self.syn_x * k[:, i]) *
                                          torch.exp(-k[:, i]))
            # # forward Euler
            # self.syn_x = self.syn_x + (1 - self.syn_x) / self.syn_tau - \
            #     self.syn_u * self.syn_x * inputs[:, i]
            # # clamp between [0,1]
            # self.syn_x = torch.clamp(self.syn_x, min=0, max=1)

            syn_x_list.append(self.syn_x)

        # concatenate original inputs and depressed inputs
        input_syn = torch.stack(syn_x_list, dim=1)
        inputs = torch.cat((inputs, input_syn * inputs), dim=2)

        hidden = []
        for i in range(inputs.shape[1]):
            x = inputs[:, i, :]
            # batch the computations into a single matrix multiplication
            self.hidden = x @ self.weight_ih + self.hidden @ self.weight_hh + self.bias
            self.hidden = F.relu(self.hidden)
            if self.noise:
                self.hidden = F.relu(self.noise(self.hidden))

            hidden.append(self.hidden)

        # update hidden layer
        hidden = torch.cat(hidden, dim=0)
        hidden = hidden.transpose(0, 1).contiguous()
        output = self.linear(hidden)

        return output, hidden, inputs,   # input_syn
