from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from common.networks.base import NNBase
from torch import Tensor
import torch as T
from common.utils import get_device
import os


class NNWrapper(ABC):
    '''
    Wrappes a neural network, with basic functionalities,
    like predict , save and load checkpoints.
    '''

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        takes a numpy array observation and returns an action probabilities
        and evaluation
        '''

    @abstractmethod
    def save_check_point(self, folder: Optional[str] = None, file: Optional[str] = None) -> None:
        '''
        Saves a checkpoint into a file.
        '''

    @abstractmethod
    def load_check_point(self, folder: Optional[str] = None, file: Optional[str] = None) -> None:
        '''
        Loads a checkpoint from a file.
        '''
    @property
    @abstractmethod
    def nn(self) -> NNBase:
        ''''''


class TorchWrapper(NNWrapper):
    '''
    Wrapper that supports Pytorch NN
    can predict , load checkpoint
    and save checkpoints.
    '''

    def __init__(self, nnet: NNBase) -> None:
        super().__init__()
        self._nn = nnet

    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs: Tensor
        wdl: Tensor
        self._nn.eval()
        observation_t = T.tensor(
            np.array([observation]), dtype=T.float32, device=get_device())
        with T.no_grad():
            probs, wdl = self._nn(observation_t)
        probs_ar = probs.data.cpu().numpy()[0]
        wdl_ar: np.ndarray = wdl.data.cpu().numpy()[0]
        return probs_ar, wdl_ar

    @property
    def nn(self) -> NNBase:
        return self._nn

    def load_check_point(self, folder: Optional[str] = None, file: Optional[str] = None) -> None:
        path = None
        if folder and file:
            path = os.path.join(folder, file)
        self.nn.load_model(path)

    def save_check_point(self, folder: Optional[str] = None, file: Optional[str] = None) -> None:
        path = None
        if folder and file:
            try :
                os.makedirs(folder,exist_ok=True)
            finally:
                ...
            path = os.path.join(folder, file)
        self.nn.save_model(path)


class TrainDroplet(TorchWrapper):
    def __init__(self, nnet: NNBase, lr: float, ratio: float) -> None:
        super().__init__(nnet)
        self.lr = lr
        self.ratio = ratio
        self.base_lr = lr

    def perturb(self, other_droplet: TrainDroplet):
        state_dict = other_droplet.nn.state_dict()
        self._nn.load_state_dict(state_dict)
        randoms = np.random.randint(2, size=2)
        self.base_lr = other_droplet.base_lr
        if randoms[0]:
            self.lr = other_droplet.lr / 1.2
        else:
            self.lr = other_droplet.lr * 1.2
        if randoms[1]:
            self.ratio = other_droplet.ratio / 1.2
        else:
            self.ratio = other_droplet.ratio * 1.2
        

    def __str__(self) -> str:
        return f'Train Droplet base lr :{self.base_lr:0.2e} current lr: {self.lr:0.2e} AC Ratio:{self.ratio:0.2e}'
