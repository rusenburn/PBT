from typing import List, Union
from common.state import State
import numpy as np


class TicTacToeState(State):
    def __init__(self, observation: np.ndarray) -> None:
        super().__init__()
        self._observation = observation
        self._legal_actions: Union[np.ndarray, None] = None
        self._is_game_over = False

    @property
    def shape(self) -> tuple:
        return self._observation.shape

    @property
    def n_actions(self) -> int:
        return 9

    def get_legal_actions(self) -> np.ndarray:
        if self._legal_actions is not None:
            return self._legal_actions.copy()
        legal_actions = np.zeros((self.n_actions,), dtype=np.int32)
        for i in range(self.n_actions):
            if self._is_legal_action(i):
                legal_actions[i] = 1
        self._legal_actions = legal_actions
        return legal_actions.copy()

    def is_game_over(self) -> bool:
        player_0: int = 0
        player_1: int = 1
        self._is_game_over = self._is_winning(
            player_0) or self._is_winning(player_1) or self._is_full()
        return self._is_game_over

    def game_result(self) -> np.ndarray:
        assert self._is_game_over
        player = self._observation[2][0][0]
        other = (player + 1) % 2
        wdl = np.zeros((3,), dtype=np.float32)
        if self._is_winning(player):
            wdl[0] += 1
        elif self._is_winning(other):
            wdl[2] += 1
        else:
            wdl[1] += 1
        return wdl

    def move(self, action: int) -> State:
        player = self._observation[2][0][0]
        next_player = (player + 1) % 2
        new_obs = self._observation.copy()
        new_obs[2] = next_player
        action_row = int(action // 3)
        action_col = int(action % 3)
        new_obs[player][action_row][action_col] = 1
        return TicTacToeState(new_obs)

    def render(self) -> None:
        player = self._observation[2][0][0]
        player_rep = ''
        if player == 0:
            player_rep = 'x'
        else:
            player_rep = 'o'
        result: List[str] = []
        result.append('****************************\n')
        result.append(f'*** Player {player_rep} has to move ***\n')
        result.append('****************************\n')
        result.append('\n')
        for row in range(3):
            for col in range(3):
                if self._observation[0][row][col] == 1:
                    result.append(' x ')
                elif self._observation[1][row][col] == 1:
                    result.append(' o ')
                else:
                    result.append(' . ')
                if col == 2:
                    result.append('\n')
        result.append('\n')
        print(''.join(result))

    def to_obs(self) -> np.ndarray:
        return self._observation.copy()

    def to_short(self) -> tuple:
        player = self._observation[2][0][0]
        space: np.ndarray = self._observation[0] - self._observation[1]
        return (player, *space.flatten(),)

    def _is_legal_action(self, action: int) -> bool:
        player_0 = 0
        player_1 = 1
        action_row = int(action//3)
        action_col = int(action % 3)
        return self._observation[player_0][action_row][action_col] == 0 and self._observation[player_1][action_row][action_col] == 0

    def _is_winning(self, player: int) -> bool:
        return self._is_horizontal_win(player) or \
            self._is_vertical_win(player) or \
            self._is_forward_win(player) or \
            self._is_backward_win(player)

    def _is_full(self) -> bool:
        player_0: int = 0
        player_1: int = 1
        for row in range(3):
            for col in range(3):
                if self._observation[player_0][row][col] == 0 and self._observation[player_1][row][col] == 0:
                    return False
        return True

    def _is_vertical_win(self, player) -> bool:
        vertical = (np.sum(self._observation[player, 0, :]) == 3 or
                    np.sum(self._observation[player, 1, :]) == 3 or
                    np.sum(self._observation[player, 2, :]) == 3)
        return vertical

    def _is_horizontal_win(self, player) -> bool:
        horizontal = (np.sum(self._observation[player, :, 0]) == 3 or
                      np.sum(self._observation[player, :, 1]) == 3 or
                      np.sum(self._observation[player, :, 2]) == 3)
        return horizontal

    def _is_forward_win(self, player) -> bool:
        forward = self._observation[player, 0, 0] == 1 == self._observation[player,
                                                                            1, 1] == self._observation[player, 2, 2]
        return forward

    def _is_backward_win(self, player) -> bool:
        backward = self._observation[player, 0, 2] == 1 == self._observation[player,
                                                                             1, 1] == self._observation[player, 2, 0]
        return backward
