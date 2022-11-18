from typing import Tuple
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.functional import Tensor
import os
import torch as T


class NNBase(nn.Module, ABC):
    def __init__(self, name, checkpoint_directory) -> None:
        super().__init__()
        self._file_name: str = os.path.join(checkpoint_directory, name)

    @abstractmethod
    def forward(self,state:Tensor) -> Tuple[Tensor,Tensor]:
        pass

    def save_model(self, path: str = None) -> None:
        '''
        Saves the model into path , if path is None
        saves the model inside self._file_name
        '''
        try:
            if path:
                path = path
            else:
                path = self._file_name
            T.save(self.state_dict(), path)
        except:
            print(f'could not save nn to path')

    def load_model(self, path: str = None) -> None:
        '''
        Loads the model from a path , if the given path is None
        loads the model from self._file_name
        '''
        try:
            if path:
                path = path
            else:
                path = self._file_name
            self.load_state_dict(T.load(path))
            print(f'The nn was loaded from {path}')
        except:
            print(f'could not load nn from {path}')

