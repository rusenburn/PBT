import numpy as np
import torch as T
import math
import time

from common.state import State
from common.networks.base import NNBase
from common.utils import get_device
from .nnmcts import MctsBase

MAX_ASYNC_SIMULATIONS = 4
DEFAULT_N = 1
DEFAULT_W = -1


class AMCTS(MctsBase):
    '''
    Asynchronous Monte Carlo Tree Search ( Not a truly asynchronous ),
    visiting a new state does not immediately evaluate that state
    instead it assign it as a loss with all leading nodes,
    gives it a normal distribution probs for valid actions for now,
    then adds it to the queue with other states to be properly evaluated
    as a group
    pros:
        * great increase in performance with GPUs
    cons:
        * adds a little noise which can be considered as a pro too
        * too heavy if the device is a CPU
    '''
    def __init__(self, n_game_actions: int, nnet: NNBase,n_sims:int, duration_in_millis: float,c: float,temperature:float) -> None:
        self._nnet: NNBase = nnet
        self._c = c
        self._n_game_actions = n_game_actions
        self._n_sims:int = n_sims
        self._duration_in_millis = duration_in_millis
        self._temperature = temperature
        self._root: State|None = None
        self._states: set[tuple] = set()
        self._states_actions: dict[tuple, list[State | None]] = dict()
        self._ns: dict[tuple, int] = dict()
        self._nsa: dict[tuple, np.ndarray] = dict()
        self._wsa: dict[tuple, np.ndarray] = dict()
        self._psa: dict[tuple, np.ndarray] = dict()
        self._actions_legality: dict[tuple, np.ndarray] = dict()
        self._rollouts: list[tuple[State, list[tuple[tuple, int]]]] = []

    def search(self,state:State)->np.ndarray:
        self._root = state
        return self._search_root(self._n_sims,self._duration_in_millis,self._temperature)
        ...
    def simulate(self, state: State, visited_path: list[tuple[tuple, int]] = None):
        if visited_path is None:
            visited_path = []
        if state.is_game_over():
            wdl = state.game_result()
            # wins - losses
            z = wdl[0] - wdl[2]
            self._backprop(visited_path, -z)
            return

        short: tuple = state.to_short()
        if short not in self._states:
            # does not exist yet , then Expand it
            self._expand_state(state, short)
            self._add_to_rollouts(state, visited_path)
            return

        best_action: int = self._find_best_action(short)
        if self._states_actions[short][best_action] is None:
            self._states_actions[short][best_action] = state.move(best_action)

        new_state: State | None = self._states_actions[short][best_action]
        if (new_state is None):
            raise ValueError()
        visited_path.append((short, best_action))
        self.simulate(new_state, visited_path)
        self._nsa[short][best_action] += DEFAULT_N
        self._ns[short] += DEFAULT_N
        self._wsa[short][best_action] += DEFAULT_W

    def _expand_state(self, state: State, short: tuple) -> None:
        assert not state.is_game_over()
        self._states.add(short)
        actions_legality = state.get_legal_actions()
        self._actions_legality[short] = actions_legality
        self._ns[short] = 0
        self._nsa[short] = np.zeros((self._n_game_actions,), dtype=np.float32)
        self._wsa[short] = np.zeros((self._n_game_actions,), dtype=np.float32)
        self._states_actions[short] = [
            None for _ in range(self._n_game_actions)]
        a = actions_legality.astype(dtype=np.float32)
        psa: np.ndarray = a/a.sum(keepdims=True)
        self._psa[short] = psa

    def _add_to_rollouts(self, state: State, visited_path: list[tuple[tuple, int]]):
        self._rollouts.append((state, visited_path))

    def _backprop(self, visited_path: list[tuple[tuple, int]], score: float):
        while (len(visited_path)) != 0:
            short, action = visited_path.pop()
            self._ns[short] += 1-DEFAULT_N
            self._nsa[short][action] += 1-DEFAULT_N
            self._wsa[short][action] += score - DEFAULT_W
            score = -score

    def _find_best_action(self, short: tuple) -> int:
        max_u, best_action = -float("inf"), -1
        wsa_ar = self._wsa[short]
        nsa_ar = self._nsa[short]
        psa_ar = self._psa[short]
        ns = self._ns[short]
        actions_legality = self._actions_legality[short]

        for action, is_legal in enumerate(actions_legality):
            if not is_legal:
                continue
            psa: float = psa_ar[action]
            nsa: int = nsa_ar[action]
            qsa: float = 0.0
            if (nsa > 0):
                qsa = wsa_ar[action]/(nsa+1e-8)
            u = qsa + self._c * psa * math.sqrt(ns)/(1 + nsa)
            if u > max_u:
                max_u = u
                best_action = action
        if best_action == -1:
            # should not happen
            # pick random action from legal actions
            best_actions = np.array(np.argwhere(
                actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
        return best_action

    def _roll(self):
        if len(self._rollouts) == 0:
            return

        self._nnet.eval()

        states: list[tuple[State, int]] = [
            (t[0], id_) for id_, t in enumerate(self._rollouts)]

        obs_ar = np.array([s.to_obs()
                           for s, _ in states], dtype=np.float32)
        obs_t = T.tensor(obs_ar, dtype=T.float32, device=get_device())
        probs_t: T.Tensor
        wdl_t: T.Tensor
        with T.no_grad():
            probs_t, wdl_t = self._nnet(obs_t)

        wdl_ar: np.ndarray = wdl_t.cpu().numpy()

        for i, wdl in enumerate(wdl_ar):
            visited_path = self._rollouts[i][1]
            z = wdl[0]-wdl[2]
            self._backprop(visited_path, -z)

        probs_ar: np.ndarray = probs_t.cpu().numpy()
        prob: np.ndarray
        for (i, prob) in enumerate(probs_ar):
            state = self._rollouts[i][0]
            short = state.to_short()
            actions_legality = self._actions_legality[short]
            assert not np.any(actions_legality != state.get_legal_actions())
            prob = prob * actions_legality
            prob = prob / prob.sum(keepdims=True)
            self._psa[short] = prob

        self._rollouts.clear()

    def _search_root(self, n_sims: int, duration_in_millis: float, temperature: float = 1.0) -> np.ndarray:
        assert self._root is not None
        duration_in_seconds = duration_in_millis/1000
        t_start = time.perf_counter()
        t_end = t_start + duration_in_seconds
        i = 0
        for i in range(n_sims):
            self.simulate(self._root)
            if i % MAX_ASYNC_SIMULATIONS == MAX_ASYNC_SIMULATIONS-1:
                self._roll()

        while (t_end > time.perf_counter()):
            self.simulate(self._root)
            if i % MAX_ASYNC_SIMULATIONS == MAX_ASYNC_SIMULATIONS - 1:
                self._roll()
            i += 1

        self._roll()
        return self._get_probs(temperature)

    def _get_probs(self, temperature: float) -> np.ndarray:
        assert self._root is not None
        short = self._root.to_short()
        action_visits = self._nsa[short]
        if temperature == 0:
            max_action_visits = np.max(action_visits)
            best_actions = np.array(np.argwhere(
                action_visits == max_action_visits)).flatten()
            best_action = np.random.choice(best_actions)

            probs: np.ndarray = np.zeros(
                (len(action_visits),), dtype=np.float32)
            probs[best_action] = 1
            return probs

        probs_with_temperature = action_visits.astype(
            np.float32)**(1/temperature)
        probs = probs_with_temperature/probs_with_temperature.sum()
        return probs
