from typing import Tuple
from common.state  import State
from common.game import Game
from tictactoe.state import TicTacToeState
import numpy as np

class TicTacToeGame(Game):
    def __init__(self) -> None:
        super().__init__()
        self._initialize_state()

    @property
    def observation_space(self) -> tuple:
        return self._state.shape

    def _initialize_state(self)->None:
        self._state = TicTacToeState(np.zeros((3, 3, 3), dtype=np.int32))
    
    def step(self, action) -> Tuple[State, bool]:
        new_state = self._state.move(action)
        done = new_state.is_game_over()
        self._state = new_state
        return new_state ,done
    
    def render(self) -> None:
        return super().render()