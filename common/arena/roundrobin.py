import concurrent.futures
from threading import Lock
from typing import Callable, List, Sequence, Tuple

from common.game import Game
from common.arena.players import PlayerBase
from common.arena.match import Match
import numpy as np
from tqdm import tqdm


class RoundRobin():
    def __init__(self, game_fn: Callable[[],Game], players: Sequence[PlayerBase], n_sets: int, render=False) -> None:
        self.game_fn = game_fn

        # TODO delete game if it is not needed
        self.game = game_fn()
        self.players = players
        self.n_sets = n_sets
        self.render = render
        self.lock = Lock()
        self.results = np.zeros((len(self.players), 3), dtype=np.int32)

    def start(self,print_progress=False) -> Tuple[np.ndarray, np.ndarray]:
        n_players = len(self.players)
        executes :List[Tuple[int,int]]= []
        for i,_ in enumerate(self.players,):
            for j in range(i, len(self.players)):
                if i == j:
                    continue
                executes.append((i,j))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    a = tqdm(executor.map(self.execute_match, executes),desc='Tournament')

        if print_progress:
            [x for x in a]
        # the number of wins is equal to the number of losses
        assert np.sum(self.results[:,0]) == np.sum(self.results[:,-1])

        # assert that result has the correct sum
        s_expected = n_players * (n_players-1) / 2 * self.n_sets * 2 # the last 2 because each game adds 1 win and 1 loss or 2 draws
        s_actual = np.sum(self.results)
        assert s_expected == s_actual

        # TODO return proper rankings
        rankings = self._get_rankings()
        return self.results, rankings

    def execute_match(self,args):
        p1_idx:int = args[0]
        p2_idx:int = args[1]
        p1 = self.players[p1_idx]
        p2 = self.players[p2_idx]
        m = Match(self.game_fn,p1,p2,self.n_sets,self.render)
        m_score = m.start()
        with self.lock:
            self.results[p1_idx]+= m_score
            self.results[p2_idx]+= m_score[::-1]
        

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

        # ranking provides the new ranking ordered accoring to the previous ranking
        # ex: if ranking = [3,2,1,0] then the player that took the 0 rank previously fell down
        # to 3 and the previously ranked 1 now has a new rank of 2 etc
        # ex: if ranking = [3,0,1,2] then the player that took the 0 rank previously fell down
        # to 3 and the previously ranked 1 now has a new rank of 0 which is good for him
        rankings = np.zeros((n_players,), dtype=np.int32)
        for i, row in enumerate(points):
            old_index = row[1]
            rankings[old_index] = i

        return rankings

    def _swap(self, points: np.ndarray, i: int, j: int) -> None:
        val = points[i].copy()
        points[i] = points[j]
        points[j] = val
