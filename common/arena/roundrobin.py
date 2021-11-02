from typing import List, Tuple

from numpy.core.numeric import indices
from common.game import Game
from arena.players import PlayerBase
from arena.match import Match
import numpy as np
import copy


class RoundRobin():
    def __init__(self, game: Game, players: List[PlayerBase], n_sets: int, render=False) -> None:
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
        points[:, 1] = np.arange(n_players) # saving the old index of players
        for i, r in enumerate(self.results):
            points[i, 0] = r[0] - r[2]

        # insertion sort
        for i in range(len(points)):
            val_1 = points[i, 0]
            for j in range(i, len(points)):
                val_2 = points[j, 0]
                if val_2 > val_1:
                    self._swap(points, i, j)
        rankings = np.zeros((n_players,),dtype=np.int32)
        for i,row in enumerate(points):
            old_index = row[1]
            rankings[old_index] = i
        
        return rankings

    def _swap(self, points: np.ndarray, i: int, j: int) -> None:
        val = points[i]
        points[i] = points[j]
        points[j] = val
