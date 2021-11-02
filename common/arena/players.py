from abc import ABC,abstractmethod
from common.state import State
from common.game import Game
from common.network_wrapper import NNWrapper
from common.nnmcts import NNMCTS
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


class NNMCTSPlayer(PlayerBase):
    def __init__(self, game: Game, wrapper: NNWrapper, n_sims: int, temperature=0.5) -> None:
        super().__init__()
        self.game = game
        self.wrapper = wrapper
        self.n_sims = n_sims
        self.temperature = temperature

    def choose_action(self, state: State) -> int:
        mcts = NNMCTS(self.game, self.wrapper, self.n_sims)
        probs = mcts.get_probs(state, self.temperature)
        a = np.random.choice(len(probs), p=probs)
        return a
