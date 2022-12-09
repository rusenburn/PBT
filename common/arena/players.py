from abc import ABC,abstractmethod
from time import time,perf_counter

from typing import Callable
from common.evaluators import Evaluator
from common.state import State
from common.game import Game
from common.network_wrapper import NNWrapper
from common.nnmcts import NNMCTS, NNMCTS2
from common.amcts import AMCTS

import numpy as np


class PlayerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def choose_action(self, state: State) -> int:
        '''
        Takes a state and returns an action
        '''


class Human(PlayerBase):
    def __init__(self) -> None:
        super().__init__()

    def choose_action(self, state: State) -> int:
        state.render()
        a = int(input('Choose Action \n'))
        return a

class RandomActionPlayer(PlayerBase):
    def __init__(self) -> None:
        super().__init__()
    
    def choose_action(self, state: State) -> int:
        actions_legality = state.get_legal_actions().astype(np.float32)
        probs = actions_legality / actions_legality.sum()
        a = np.random.choice(len(actions_legality),p=probs)
        return a


class NNMCTSPlayer(PlayerBase):
    def __init__(self, n_game_actions, wrapper: NNWrapper, n_sims: int, temperature=0.5) -> None:
        super().__init__()
        self.n_game_actions =n_game_actions
        self.wrapper = wrapper
        self.n_sims = n_sims
        self.temperature = temperature

    def choose_action(self, state: State) -> int:
        mcts = NNMCTS(self.n_game_actions, self.wrapper, self.n_sims,temperature=self.temperature)
        t_start = perf_counter()
        probs = mcts.search(state)
        a = np.random.choice(len(probs), p=probs)
        duration = perf_counter() - t_start
        # print(f"sims per second\t  {self.n_sims/duration:0.2f}")
        return a

class NNMCTS2Player(PlayerBase):
    def __init__(self, n_game_actions:int, evaluator: Evaluator, n_sims: int, temperature=0.5) -> None:
        super().__init__()
        self.n_game_actions =n_game_actions
        self.evaluator = evaluator
        self.n_sims = n_sims
        self.temperature = temperature

    def choose_action(self, state: State) -> int:
        mcts = NNMCTS2(self.n_game_actions, self.evaluator, self.n_sims,temperature=self.temperature)
        t_start = perf_counter()
        probs = mcts.search(state)
        a = np.random.choice(len(probs), p=probs)
        duration = perf_counter() - t_start
        # print(f"sims per second\t  {self.n_sims/duration:0.2f}")
        return a


class AMCTSPlayer(PlayerBase):
    def __init__(self,n_game_actions:int,nn_wrapper:NNWrapper,n_sims:int,temperature=0.5) -> None:
        super().__init__()
        self._n_game_actions:int = n_game_actions
        self._nn_wrapper = nn_wrapper
        self._n_sims = n_sims
        self._temperature = temperature
    
    def choose_action(self, state: State) -> int:
        mcts = AMCTS(self._n_game_actions,self._nn_wrapper.nn,self._n_sims,0,c=1.0,temperature=self._temperature)
        t_start = perf_counter()
        probs = mcts.search(state)
        a = np.random.choice(self._n_game_actions,p=probs)
        duration = perf_counter() - t_start
        # print(f"sims per second async\t  {self._n_sims/duration:0.2f}")
        return a




# class DualNNMCTSPlayer(PlayerBase):
#     '''
#     Used for debug purposes only to check 
#     the difference between old NNMCTS and NNMCTS2
#     '''
#     def __init__(self, n_game_actions, wrapper: NNWrapper, n_sims: int, temperature=0.5) -> None:
#         super().__init__()
#         self.n_game_actions =n_game_actions
#         self.wrapper = wrapper
#         self.n_sims = n_sims
#         self.temperature = temperature

#     def choose_action(self, state: State) -> int:
#         mcts_1 = NNMCTS(self.n_game_actions, self.wrapper, self.n_sims)
#         probs_1 = mcts_1.get_probs(state, self.temperature)

#         mcts_2 = NNMCTS2(self.n_game_actions, self.wrapper, self.n_sims)
#         probs_2 = mcts_2.get_probs(state, self.temperature)

#         probs_diff = np.zeros((self.n_game_actions,2),dtype=np.float32)
#         probs_diff[:,0] = probs_1
#         probs_diff[:,1] = probs_2
#         diff = probs_1-probs_2
#         a = np.random.choice(len(probs_1), p=probs_1)
#         return a