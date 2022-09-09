from typing import List
from common.state import State
import numpy as np

class ConnectFourState(State):
    def __init__(self,observation:np.ndarray,last_action:int,turn:int) -> None:
        super().__init__()
        self._observation = observation
        self._legal_actions : np.ndarray|None = None
        self._is_game_over : bool|None = None
        self._turn = turn
        self._last_action = last_action
        self._short : None|tuple = None
    
    @property
    def shape(self)->tuple:
        return self._observation.shape
    
    @property
    def n_actions(self)->int:
        return 7
    
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
        if self._is_game_over is not None:
            return self._is_game_over
        if self._last_action<0:
            return False
        player :int = self._observation[2][0][0]
        other : int = 1-player
        if self._is_winning(other) or self._is_full():
            self._is_game_over = True
        else :
            self._is_game_over = False
        return self._is_game_over
    
    def game_result(self) -> np.ndarray:
        player = self._observation[2][0][0]
        other = 1-player
        wdl = np.zeros((3,), dtype=np.int32)
        if self._is_winning(player):
            wdl[0] += 1
        elif self._is_winning(other):
            wdl[2] += 1
        else:
            wdl[1] += 1
        return wdl.copy()
    
    def move(self, action: int) -> State:
        legal_actions = self.get_legal_actions()
        if legal_actions[action] == 0:
            raise ValueError(f'action {action} is not allowed')
        player = self._observation[2][0][0]
        next_player = (player + 1) % 2
        new_obs = self._observation.copy()
        new_obs[2] = next_player
        col = action
        for row in range(6):
            if new_obs[player][row][col] == 0 and new_obs[next_player][row][col] == 0:
                break
        new_obs[player][row][col]=1
        return ConnectFourState(new_obs,action,self._turn+1)
        
    def render(self) -> None:
        string_list: List[str] = []
        player = self._observation[2][0][0]
        player_rep = ''
        if player == 0:
            player_rep = 'X'
        else:
            player_rep = 'O'
        string_list.append('****************************\n')
        string_list.append(f'*** Player {player_rep} has to move ***\n')
        string_list.append('****************************\n')
        for row in range(6):
            string_list.append('\n')
            string_list.append('____' * 7)
            string_list.append('\n')
            for col in range(7):
                string_list.append('|')
                if self._observation[0][6-row-1][col] == 1:
                    string_list.append(' X ')
                elif self._observation[1][6-row-1][col] == 1:
                    string_list.append(' O ')
                else:
                    string_list.append('   ')
                if col == 6:  # 0-index last column
                    string_list.append('|')
            if row == 5:  # 0-index last row
                string_list.append('\n')
                string_list.append('----' * 7)
        string_list.append('\n')
        for i in range(7):
            string_list.append(f'  {i} ')
        print("".join(string_list))
        
    

    def to_obs(self) -> np.ndarray:
        return self._observation.copy()
    
    
    def to_short(self) -> tuple:
        if self._short is not None:
            return self._short
        player = self._observation[2][0][0]
        space: np.ndarray = self._observation[0] - self._observation[1]
        self._short = (player, *space.copy().flatten(),)
        return self._short

    def get_symmetries(self, probs: np.ndarray) -> List[tuple[State, np.ndarray]]:
        obs_1 = self._observation[:,:,::-1]
        sym_last_action = 6-self._last_action
        sym_state_1 = ConnectFourState(obs_1,sym_last_action,self._turn)
        sym_probs_1 = probs[::-1]
        return [(sym_state_1,sym_probs_1)]
        ...
    def _is_winning(self,player:int)->bool:
        if self._last_action < 0:
            return False
        other = 1-player
        row: int
        rows_count = 6
        for row in range(rows_count):
            if row+1 == rows_count or (
                    self._observation[player][row+1][self._last_action] == 0 and
                    self._observation[other][row+1][self._last_action] == 0):
                break

        if self._is_vertical_win(player, row, self._last_action):
            return True
        if self._is_horizontal_win(player, row, self._last_action):
            return True
        if self._is_forward_diagonal_win(player, row, self._last_action):
            return True
        if self._is_backward_diagonal_win(player, row, self._last_action):
            return True
        return False

    def _is_full(self)->bool:
        game_rows = 6
        game_cols = 7
        game_blocks = game_rows * game_cols
        return self._turn == game_blocks - 1  # 0-based
    
    
    def _is_vertical_win(self, player, row, col) -> bool:
        count = 1
        current_col = col+1
        game_cols = 7
        while current_col < game_cols and self._observation[player][row][current_col]:
            count += 1
            current_col += 1

        current_col = col-1
        while current_col >= 0 and self._observation[player][row][current_col]:
            count += 1
            current_col -= 1
        return count >= 4

    def _is_horizontal_win(self, player, row, col) -> bool:
        count = 1
        current_row = row+1
        game_rows = 6
        while current_row < game_rows and self._observation[player][current_row][col]:
            count += 1
            current_row += 1
        current_row = row-1
        while current_row >= 0 and self._observation[player][current_row][col]:
            count += 1
            current_row -= 1
        return count >= 4

    def _is_forward_diagonal_win(self, player, row, col) -> bool:
        count = 1
        i = 1
        game_rows = 6
        game_cols = 7
        while row+i < game_rows and col+i < game_cols and self._observation[player][row+i][col+i]:
            count += 1
            i += 1
        i = 1
        while row-i >= 0 and col-i >= 0 and self._observation[player][row-i][col-i]:
            count += 1
            i += 1
        return count >= 4

    def _is_backward_diagonal_win(self, player, row, col) -> bool:
        count = 1
        i = 1
        game_rows = 6
        game_cols = 7
        while row+i < game_rows and col-i >= 0 and self._observation[player][row+i][col-i]:
            count += 1
            i += 1
        i = 1
        while row-i >= 0 and col+i < game_cols and self._observation[player][row-i][col+i]:
            count += 1
            i += 1
        return count >= 4
    
    def _is_legal_action(self, col: int) -> bool:
        last_col_idx = 6
        if col > last_col_idx or col < 0:
            return False
        player: int = self._observation[2][0][0]
        other: int = 1-player
        last_row = 5
        # check if the top cell in the col is empty
        result = self._observation[player][last_row][col] == 0 and self._observation[other][last_row][col] == 0
        return result
