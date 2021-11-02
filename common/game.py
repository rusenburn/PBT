from abc import abstractmethod,ABC
from typing import Tuple
from common.state import State


class Game(ABC):
    def __init__(self)->None:
        super().__init__()
    
    @property
    @abstractmethod
    def n_actions(self)->int:
        pass

    
    @property
    @abstractmethod
    def observation_space(self)->tuple:
        pass
    
    @abstractmethod
    def reset(self)->State:
        pass
    
    @abstractmethod
    def step(self,action)->Tuple[State,bool]:
        pass

    @abstractmethod
    def render(self)->None:
        pass