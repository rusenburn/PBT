from typing import Dict, List, Set
from common.game import Game
from common.state import State
from common.network_wrapper import NNWrapper
import numpy as np
import math


class NNMCTS():
    def __init__(self, game: Game, nnet: NNWrapper, n_sims: int, cpuct=1) -> None:
        self.game: Game = game
        self.nnet = nnet
        self.n_sims = n_sims
        self.cpuct = cpuct
        self.visited: Set[tuple] = set()
        # self.qsa: Dict[tuple, float] = {}
        self.nsa: Dict[tuple, int] = {}
        self.ns: Dict[tuple, int] = {}
        self.ps: Dict[tuple, np.ndarray] = {}
        self.wsa: Dict[tuple, int] = {}  # Wins
        self.dsa: Dict[tuple, int] = {}  # draws
        self.lsa: Dict[tuple, int] = {}  # losses

    def search(self, state: State) -> np.ndarray:
        return self._search(state, 0)

    def _search(self, state: State, depth: int) -> np.ndarray:
        if state.is_game_over():
            wdl = state.game_result()
            return wdl[::-1]

        s = state.to_short()
        if s not in self.visited:
            self.visited.add(s)
            obs = state.to_obs()
            self.ps[s], wdl = self.nnet.predict(obs)
            return wdl[::-1]
        max_u: float
        best_a: int
        max_u,  best_a = -float("inf"), -1
        legal_actions = state.get_legal_actions()
        # for a, is_legal in enumerate(legal_actions):
        #     if not is_legal:
        #         continue
        #     if (*s, a) not in self.nsa:
        #         self.nsa[(*s, a)] = 0
        #     if s not in self.ns:
        #         self.ns[s] = 0
        #     if (*s, a) in self.wsa:
        #         wsa = self.wsa[(*s, a)]
        #         lsa = self.lsa[(*s, a)]
        #         nsa = self.nsa[(*s, a)]
        #         qsa = (wsa - lsa) / nsa
        #         u = qsa + self.cpuct * self.ps[s][a] * \
        #             math.sqrt(self.ns[s]) / (1 + nsa)
        #     else:
        #         u = self.cpuct * \
        #             self.ps[s][a] * \
        #             math.sqrt(self.ns[s]) / (1 + self.nsa[(*s, a)])

        #     if u > max_u:
        #         max_u = u
        #         best_a = a

        # if best_a == -1:
        #     # print(
        #     #     f'Warning :: a monte carlo tree search gave a non existing best action of {best_a}.',end="")
        #     best_actions = np.array(np.argwhere(
        #         legal_actions == 1)).flatten()
        #     best_action = np.random.choice(best_actions)
        #     best_a = best_action
        # a = best_a
        # new_state = state.move(a)
        # wdl = self._search(new_state, depth=depth+1)
        # if (*s, a) in self.wsa:
        #     self.wsa[(*s, a)] += wdl[0]
        #     self.dsa[(*s, a)] += wdl[1]
        #     self.lsa[(*s, a)] += wdl[2]
        #     self.nsa[(*s, a)] += 1
        # else:
        #     self.wsa[(*s, a)] = wdl[0]
        #     self.dsa[(*s, a)] = wdl[1]
        #     self.lsa[(*s, a)] = wdl[2]
        #     self.nsa[(*s, a)] = 1
        # self.ns[s] += 1
        # return wdl[::-1]
        for a, is_legal in enumerate(legal_actions):
            if not is_legal:
                continue
            if (s, a) not in self.nsa:
                self.nsa[(s, a)] = 0
            if s not in self.ns:
                self.ns[s] = 0
            if (s, a) in self.wsa:
                wsa = self.wsa[(s, a)]
                lsa = self.lsa[(s, a)]
                nsa = self.nsa[(s, a)]
                qsa = (wsa - lsa) / nsa
                u = qsa + self.cpuct * self.ps[s][a] * \
                    math.sqrt(self.ns[s]) / (1 + nsa)
            else:
                u = self.cpuct * \
                    self.ps[s][a] * \
                    math.sqrt(self.ns[s]) / (1 + self.nsa[(s, a)])

            if u > max_u:
                max_u = u
                best_a = a

        if best_a == -1:
            # print(
            #     f'Warning :: a monte carlo tree search gave a non existing best action of {best_a}.',end="")
            best_actions = np.array(np.argwhere(
                legal_actions == 1)).flatten()
            best_action = np.random.choice(best_actions)
            best_a = best_action
        a = best_a
        new_state = state.move(a)
        wdl = self._search(new_state, depth=depth+1)
        if (s, a) in self.wsa:
            self.wsa[(s, a)] += wdl[0]
            self.dsa[(s, a)] += wdl[1]
            self.lsa[(s, a)] += wdl[2]
            self.nsa[(s, a)] += 1
        else:
            self.wsa[(s, a)] = wdl[0]
            self.dsa[(s, a)] = wdl[1]
            self.lsa[(s, a)] = wdl[2]
            self.nsa[(s, a)] = 1
        self.ns[s] += 1
        return wdl[::-1]

    def get_probs(self, state: State, temperature: float = 1)->np.ndarray:
        assert not state.is_game_over()
        for _ in range(self.n_sims):
            self._search(state, 0)

        s = state.to_short()

        # counts: List[float] = [self.nsa[(*s, a)] if(*s, a)
        #                        in self.nsa else 0 for a in range(self.game.n_actions)]

        counts: List[float] = [self.nsa[(s, a)] if(s, a)
                               in self.nsa else 0 for a in range(self.game.n_actions)]

        if temperature == 0:
            best_actions = np.array(np.argwhere(
                counts == np.max(counts))).flatten()
            best_action = np.random.choice(best_actions)
            probs:np.ndarray = np.zeros((len(counts),),dtype=np.float32)
            # probs: List[float] = [0] * len(counts)
            probs[best_action] = 1
            return probs

        counts = [x**(1/temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = np.array([x/counts_sum for x in counts],dtype=np.float32)
        return probs
