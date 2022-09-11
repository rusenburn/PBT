import numpy as np

from common.game import Game
from typing import Tuple,List
from .state import OthelloState

class OthelloGame(Game):
    def __init__(self) -> None:
        self._state = self._get_new_state()

    
    @property
    def n_actions(self) -> int:
        return 65
    
    @property
    def observation_space(self) -> tuple:
        return self._state.shape


    def reset(self) -> 'OthelloState':
        self._state = self._get_new_state()
        return self._state
    
    def step(self, action) -> Tuple['OthelloState', bool]:
        new_state  =self._state.move(action)
        done = new_state.is_game_over()
        self._state = new_state
        return new_state,done
    
    def render(self):
        self._state.render()
    
    def _get_new_state(self)->OthelloState:
        game_rows = 8
        game_cols = 8
        players =2
        obs = np.zeros((players+1,game_rows,game_cols),dtype=np.int32)
        obs[1,3,3]=1
        obs[1,4,4]=1
        obs[0,3,4]=1
        obs[0,4,3]=1
        n_consecutive_skips = 0
        return OthelloState(obs,n_consecutive_skips)