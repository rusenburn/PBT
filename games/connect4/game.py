from typing import Tuple
import numpy as np
from common.game import Game
from common.state import State
from .state import ConnectFourState

class ConnectFourGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self._state = self._initialize_state()
    
    @property
    def observation_space(self) -> tuple:
        return self._state.shape
    @property
    def n_actions(self)->int:
        return 7

    def reset(self)->ConnectFourState:
        self._state = self._initialize_state()
        return self._state
    
    def step(self, action) -> Tuple[State, bool]:
        new_state = self._state.move(action)
        done = new_state.is_game_over()
        self._state = new_state
        return new_state,done

    def render(self) -> None:
        self._state.render()

    def _initialize_state(self):
        game_rows = 6
        game_cols = 7
        players = 2
        obs = np.zeros((players+1, game_rows, game_cols), dtype=np.int)
        turn = 0
        last_action = -1
        return  ConnectFourState(obs, last_action, turn)