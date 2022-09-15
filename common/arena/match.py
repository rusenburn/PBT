from typing import Callable
from common.game import Game
from common.arena.players import PlayerBase
import numpy as np
from common.state import State


class Match():
    def __init__(self, game_fn: Callable[[],Game], player_1: PlayerBase, player_2: PlayerBase, n_sets=1, render=False) -> None:
        self.player_1 = player_1
        self.player_2 = player_2
        self.game_fn = game_fn
        self.game = game_fn()
        self.n_sets = n_sets
        self.render = render
        self.scores = np.zeros((3,), dtype=np.int32)  # W - D - L for player_1

    def start(self) -> np.ndarray:
        starting_player = 0
        for _ in range(self.n_sets):
            scores = self._play_set(starting_player)
            self.scores += scores
            starting_player = 1-starting_player
        return self.scores

    def _play_set(self, starting_player) -> np.ndarray:
        players = [self.player_1, self.player_2]
        state = self.game.reset()
        done = False
        current_player = starting_player
        while True:
            if self.render:
                state.render()
            player = players[current_player]
            a = player.choose_action(state)
            legal_actions = state.get_legal_actions()
            if not legal_actions[a]:
                print(f'player {current_player+1} chose wrong action {a}\n')
                continue
            new_state: State
            new_state, done = self.game.step(a)
            state = new_state
            current_player = 1-current_player
            if done:
                assert new_state.is_game_over()
                result: np.ndarray = new_state.game_result()
                if current_player != 0:
                    result = result[::-1]
                break
        if self.render:
            state.render()
        return result
