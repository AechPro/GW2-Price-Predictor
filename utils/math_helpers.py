import scipy.signal
import numpy as np
import torch
import torch.nn as nn
from functools import partial


def compute_torch_normal_entropy(sigma):
    return (0.5 + 0.5*np.log(2*np.pi) + torch.log(sigma)).sum(dim=-1).item()


def minmax_norm(x, min_val, max_val):
    if min_val == max_val:
        return x
    return (x - min_val) / (max_val - min_val)

def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()

def compute_array_stats(arr):
    if len(arr) == 0 or type(arr) not in (list, tuple, np.ndarray):
        return 0,1,0,0
    return np.mean(arr), np.std(arr), np.min(arr), np.max(arr)


def compute_discounted_future_sum(arr, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], arr[::-1], axis=0)[::-1]


def apply_affine_map(value, from_min, from_max, to_min, to_max):
        if from_max == from_min or to_max == to_min:
            return to_min

        mapped = (value - from_min) * (to_max - to_min) / (from_max - from_min)
        mapped += to_min

        return mapped


def map_policy_to_continuous_action(policy_output):
    n = policy_output.shape[-1]//2
    if len(policy_output.shape) == 1:
        mean = policy_output[:n]
        std = policy_output[n:]

    else:
        mean = policy_output[:, :n]
        std = policy_output[:, n:]

    std = apply_affine_map(std, -1, 1, 1e-1, 1)
    return mean, std

def orthogonal_initialization(model):
        """
        Orthogonal initialization procedure.
        :return:
        """

        def init_weights_orthogonal(module: nn.Module, gain: float = 1) -> None:
            """
            COPIED FROM STABLE-BASELINES3
            Orthogonal initialization (used in PPO and A2C)
            """
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

        model_gain = 1.0
        action_gain = 10.0

        n_trainable_layers = 0
        for p in model.parameters():
            trainable = False
            if p.requires_grad:
                trainable = True
            if trainable:
                n_trainable_layers += 1

        i = 0
        for layer in model:
            trainable = False
            for p in layer.parameters():
                if p.requires_grad:
                    trainable = True
                    break
            if trainable:
                if i < n_trainable_layers - 1:
                    layer.apply(partial(init_weights_orthogonal, gain=model_gain))
                else:
                    layer.apply(partial(init_weights_orthogonal, gain=action_gain))
                i += 1

def round_list(lst, sig_figs=3):
    return [round(arg, sig_figs) for arg in lst]

class SigmoidDistribution(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        a, b, c = x.shape
        sigm = self.sigm(x)
        s = sigm.sum(dim=-1).view(a, b, 1)
        norm = sigm / (1e-12 + s)
        output = torch.where(s > 1, norm, sigm)
        return output