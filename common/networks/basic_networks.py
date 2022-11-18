from typing import Tuple
import torch.nn as nn
from torch import Tensor
from common.utils import get_device
from common.networks.base import NNBase
import numpy as np
import torch as T

class SharedResNetwork(NNBase):
    def __init__(self,
                 shape: tuple,
                 n_actions: int,
                 name='shared_res_network',
                 checkpoint_directory='tmp',
                 filters=128,
                 fc_dims=512,
                 n_blocks=3):

        super().__init__(name, checkpoint_directory)

        self._blocks = nn.ModuleList(
            [ResBlock(filters) for _ in range(n_blocks)])

        self._shared = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1),
            *self._blocks)

        self._pi_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))

        self._wdl_head = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, 3))
            
        device = get_device()
        self._blocks.to(device)
        self._shared.to(device)
        self._pi_head.to(device)
        self._wdl_head.to(device)

    def forward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        shared: Tensor = self._shared(state)
        pi: Tensor = self._pi_head(shared)
        wdl_nums: Tensor = self._wdl_head(shared)
        probs: Tensor = pi.softmax(dim=-1)
        wdl_probs = wdl_nums.softmax(dim=-1)
        return probs, wdl_probs


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self._block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self._se = SqueezeAndExcite(channels, squeeze_rate=4)

    def forward(self, state: Tensor) -> Tensor:
        initial = state
        output: Tensor = self._block(state)
        output = self._se(output, initial)
        output += initial
        output = output.relu()
        return output


class SqueezeAndExcite(nn.Module):
    def __init__(self, channels, squeeze_rate):
        super().__init__()
        self.channels = channels
        self.prepare = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self._fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, int(channels//squeeze_rate)),
            nn.ReLU(),
            nn.Linear(int(channels//squeeze_rate), channels*2)
        )

    def forward(self, state: Tensor, input_: Tensor) -> Tensor:
        shape_ = input_.shape
        prepared: Tensor = self.prepare(state)
        prepared = self._fcs(prepared)
        splitted = prepared.split(self.channels, dim=1)
        w: Tensor = splitted[0]
        b: Tensor = splitted[1]
        z = w.sigmoid()
        z = z.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        b = b.unsqueeze(-1).unsqueeze(-1).expand((-1, -
                                                  1, shape_[-2], shape_[-1]))
        output = (input_*z) + b
        return output

class RolloutPolicyNetwork(NNBase):
    def __init__(self,shape: tuple,n_actions: int) -> None:
        super().__init__(name='rollout_policy_network',
                 checkpoint_directory='tmp')
        filters = 3
        fc_dims = 128
        self._pi_head = nn.Sequential(
            nn.Conv2d(shape[0], filters, 3, 1, 1,dtype=T.float32),
            nn.Flatten(),
            nn.Linear(shape[1]*shape[2]*filters, fc_dims),
            nn.ReLU(),
            nn.Linear(fc_dims, n_actions))
        self._pi_head.to(get_device())

    def forward(self,state:Tensor):
        pi: Tensor = self._pi_head(state)
        probs: Tensor = pi.softmax(dim=-1)
        v : Tensor = T.tensor(np.zeros((state.size()[0],3)),dtype=T.float32)
        return probs, v

        