from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class State(ABC):
    def __init__(self)->None:
        super().__init__()
    
    @abstractmethod
    def get_legal_actions(self)->np.ndarray:
        '''
        Returns binary numpy array with length of the number of actions
        legal actions have a value of 1
        '''
    
    @abstractmethod
    def is_game_over(self)->bool:
        '''
        Returns True if the game is over
        or False if it is not
        '''
    
    @abstractmethod
    def game_result(self)->np.ndarray:
        '''
        Returns a numpy array
        '''
    
    @abstractmethod
    def move(self,action:int)->State:
        '''
        Returns the new state of the game after
        peforming an action
        '''
    
    @abstractmethod
    def to_obs(self)->np.ndarray:
        '''
        Converts the state into numpy array
        '''
    
    @abstractmethod
    def render(self)->None:
        '''
        Renders the current state
        '''
    
    @abstractmethod
    def to_short(self)->tuple:
        '''
        Returns short form for the current state
        can be used as a key
        '''
    @abstractmethod
    def get_symmetries(self,probs:np.ndarray)->List[tuple[State,np.ndarray]]:
        '''
        Takes State action or action probs and returns 
        List of equivalent states with provided probs
        '''
