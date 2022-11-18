import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from common.network_wrapper import NNWrapper
from common.state import State


class Evaluator(ABC):
    '''
    abstract class for monte carlo state evaluation
    '''
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def evaluate(self,state:State)->Tuple[np.ndarray,np.ndarray]:
        '''
        Takes a state and returns a tuple of two numpy arrays
        which represents policy actions probs and state evaluation
        '''

class DeepNNEvaluator(Evaluator):
    def __init__(self,wrapper:NNWrapper) -> None:
        super().__init__()
        self.wrapper = wrapper
    
    def evaluate(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        assert not state.is_game_over()
        obs = state.to_obs()
        probs , wdl = self.wrapper.predict(obs)
        return probs , wdl

class RandomRollouts(Evaluator):
    '''
    Evaluating state 
    '''
    def __init__(self) -> None:
        super().__init__()
    
    def evaluate(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Evaluating state by playing random legal moves until the game ends
        and return the score , and returns equally probs over legal moves
        returns probs , win-draw-loss score
        '''
        assert not state.is_game_over()
        player = 0
        actions_legality =  state.get_legal_actions()
        n_legal_actions = actions_legality.sum()
        probs = actions_legality.astype(np.float32) / n_legal_actions
        best_actions = np.array(np.argwhere(actions_legality == 1).flatten())
        best_action = np.random.choice(best_actions)
        best_a = best_action
        state = state.move(best_a)
        player = 1-player
        while not state.is_game_over():
            actions_legality =  state.get_legal_actions()
            best_actions = np.array(np.argwhere(actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
            best_a = best_action
            state = state.move(best_a)
            player = 1-player
        state_wdl = state.game_result()
        wdl : np.ndarray
        if player == 0:
            wdl = state_wdl
        else:
            wdl = state_wdl[::-1]
        return probs ,wdl

class RolloutPolicy(Evaluator):
    '''
    Evaluate a state by performing picking actions according to
    nnet policy until the game is over
    returning uniformally distributed legal action probs 
    and game result as an evaluation
    '''
    def __init__(self,nnet:NNWrapper) -> None:
        super().__init__()
        self.policy = nnet
    
    def evaluate(self, state: State) -> Tuple[np.ndarray, np.ndarray]:
        assert not state.is_game_over()
        player = 0
        actions_legality =  state.get_legal_actions()
        n_legal_actions = actions_legality.sum()

        # unlike normal nn evaluation we should return available actions probs with EQUAL CHANCES
        probs_result:np.ndarray = actions_legality.astype(np.float32) / n_legal_actions

        while not state.is_game_over():
            actions_legality =  state.get_legal_actions()
            step_probs,_ = self.policy.predict(state.to_obs())

            # keep the probabilities of legal action
            step_probs = step_probs * actions_legality

            # divide by their sum , so they add up to 1
            step_probs = step_probs / step_probs.sum()
            
            # pick an action randomly by its probabilty
            best_a = np.random.choice(len(step_probs),p=step_probs)

            # move state and change player
            state = state.move(best_a)
            player = 1-player
        state_wdl = state.game_result()
        wdl : np.ndarray
        if player == 0:
            wdl = state_wdl
        else:
            wdl = state_wdl[::-1]
        return probs_result ,wdl



