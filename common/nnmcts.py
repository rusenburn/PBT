from typing import Dict, List, Set
from common.state import State
from common.network_wrapper import NNWrapper
import numpy as np
import math


class NNMCTS():
    def __init__(self, n_game_actions, nnet: NNWrapper, n_sims: int, cpuct=1) -> None:
        self.n_game_actions = n_game_actions
        self.nnet = nnet
        self.n_sims = n_sims
        self.cpuct = cpuct
        self.visited: Set[tuple] = set()
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
                               in self.nsa else 0 for a in range(self.n_game_actions)]

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

class NNMCTS2():
    '''
    Just like a first NNMCTS except that it saves the states
    does not work on non deterministic env because it assumes
    that performing an action will always give the same state
    '''
    def __init__(self, n_game_actions, nnet: NNWrapper, n_sims: int, cpuct=1) -> None:
        self.n_game_actions = n_game_actions
        self.nnet = nnet
        self.n_sims = n_sims
        self.cpuct = cpuct
        self.parent_node: Node | None= None

    def search(self, state: State) -> np.ndarray:
        if self.parent_node is None or self.parent_node.state != state:
            self.parent_node = Node(state,self.n_game_actions,self.cpuct)
        return self.parent_node.search(self.nnet)

    def get_probs(self, state: State, temperature: float = 1)->np.ndarray:
        assert not state.is_game_over()
        self.parent_node = Node(state,self.n_game_actions,self.cpuct)
        return self.parent_node.search_and_get_probs(self.nnet,self.n_sims,temperature)

class Node:
    def __init__(self,state:State ,n_game_actions:int,cpuct=1) -> None:
        self.state = state
        self.actions_legality :np.ndarray|None = None
        self.cpuct = cpuct
        self.n_game_actions = n_game_actions
        self.children: List[Node|None] = [None] * n_game_actions
        self.probs :np.ndarray = np.zeros((n_game_actions,))
        self.n : int = 0
        self.na : np.ndarray = np.zeros((n_game_actions,),dtype=np.int32)
        self.wdla : np.ndarray  = np.zeros((n_game_actions,3),dtype=np.float32)
        self.wa : np.ndarray = self.wdla[:,0]
        self.da : np.ndarray = self.wdla[:,1]
        self.la : np.ndarray = self.wdla[:,2]

        #cached
        self.is_game_over:bool|None = None
        self.game_result :np.ndarray|None= None
        
    
    def search(self,nnet:NNWrapper)->np.ndarray:
        # check if we cached game over
        if self.is_game_over is None:
            self.is_game_over = self.state.is_game_over()

        # if game over return result and cache
        if self.is_game_over:
            if self.game_result is None:
                self.game_result = self.state.game_result()
                return self.game_result[::-1]
        
        if self.actions_legality is None: # first time visit
            self.actions_legality =  self.state.get_legal_actions()
            self.children = [None for action in self.actions_legality]
            probs , wdl = nnet.predict(self.state.to_obs())
            self.probs = probs
            return wdl[::-1]
        
        a = self._get_best_action()

        if self.children[a] is None:
            new_state = self.state.move(a)
            self.children[a] = Node(new_state,self.n_game_actions,self.cpuct)
        new_node = self.children[a]

        assert new_node is not None

        wdl = new_node.search(nnet)
        self.wa[a] += wdl[0]
        self.da[a] += wdl[1]
        self.la[a] += wdl[2]
        self.na[a] += 1
        self.n+=1

        return wdl[::-1]

    
    
    # Do search number of times first before doing get probs
    def get_probs(self,temperature:float=1)->np.ndarray:

        # get a copy of array of number of times an action was performed in this node during search
        action_visits :np.ndarray = self.na.copy()
        # if exploring temperature was 0 , give the best action 
        if temperature == 0:
            max_action_visits = np.max(action_visits)
            best_actions = np.array(np.argwhere(action_visits == max_action_visits)).flatten()
            best_action = np.random.choice(best_actions)
            
            probs : np.ndarray = np.zeros((len(action_visits),),dtype=np.float32)
            probs[best_action] = 1
            return probs

        # if exploring temperature was not 0 get an action depends on the number of times this action was perfomed 
        probs_with_temperature = action_visits.astype(np.float32)**(1/temperature)
        probs = probs_with_temperature/probs_with_temperature.sum()
        return probs
    
    def search_and_get_probs(self,nnet:NNWrapper,n_sims:int,temperature:float=1)->np.ndarray:
        if self.is_game_over is None:
            self.is_game_over = self.state.is_game_over()
            assert not self.is_game_over
        
        for _ in range(n_sims):
            self.search(nnet)
        
        probs = self.get_probs(temperature)
        return probs
    
    def _get_best_action(self):
        max_u,best_a = -float("inf"),-1
        for a , is_legal in enumerate(self.actions_legality):
            if not is_legal:
                continue
            
            na:int = self.na[a]
            qsa : float = 0
            if na > 0 :
                wa = self.wa[a]
                la = self.la[a]
                qsa = (wa - la) / na
            u = qsa + self.cpuct * self.probs[a] * math.sqrt(self.n) / (1 + na)

            # u = qsa + self.cpuct * self.ps[s][a] * math.sqrt(self.ns[s]) / (1 + nsa)
            if u > max_u:
                max_u = u
                best_a = a
        if best_a == -1:
            # should not happen , unless was given a very bad probabilities by nnet
            # pick random action from legal actions
            best_actions = np.array(np.argwhere(self.actions_legality == 1).flatten())
            best_action = np.random.choice(best_actions)
            best_a = best_action
        
        a = best_a
        return a