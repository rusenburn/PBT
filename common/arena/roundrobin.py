from typing import Sequence, Tuple

from common.game import Game
from common.arena.players import PlayerBase
from common.arena.match import Match
import numpy as np
import copy


class RoundRobin():
    def __init__(self, game: Game, players: Sequence[PlayerBase], n_sets: int, render=False) -> None:
        self.game = copy.deepcopy(game)
        self.players = players
        self.n_sets = n_sets
        self.render = render
        self.results = np.zeros((len(self.players), 3), dtype=np.int32)

    def start(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self.players)):
            for j in range(i, len(self.players)):
                if i == j:
                    continue
                p1 = self.players[i]
                p2 = self.players[j]
                m = Match(self.game, p1, p2, self.n_sets, self.render)
                m_score = m.start()
                self.results[i] += m_score
                self.results[j] += m_score[::-1]
        n_players = len(self.players)
        # TODO return proper rankings
        rankings = self._get_rankings()
        return self.results, rankings

    def _get_rankings(self) -> np.ndarray:
        n_players: int = len(self.players)
        points: np.ndarray = np.zeros((n_players, 2), dtype=np.int32)
        points[:, 0] = np.zeros((n_players,), dtype=np.int32)
        points[:, 1] = np.arange(n_players)  # saving the old index of players
        for i, r in enumerate(self.results):
            points[i, 0] = r[0] - r[2]

        for i in range(len(points)):
            for j in range(i, 0, -1):
                if points[j, 0] > points[j-1, 0]:
                    self._swap(points, j, j-1)
                else:
                    break

        rankings = np.zeros((n_players,), dtype=np.int32)
        for i, row in enumerate(points):
            old_index = row[1]
            rankings[old_index] = i

        return rankings

    def _swap(self, points: np.ndarray, i: int, j: int) -> None:
        val = points[i].copy()
        points[i] = points[j]
        points[j] = val
