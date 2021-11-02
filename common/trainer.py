from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from common.arena.match import Match
from torch.functional import Tensor
from common.arena.players import NNMCTSPlayer
from common.arena.roundrobin import RoundRobin
from common.network_wrapper import TrainDroplet
from common.networks.base import NNBase
from common.networks.basic_networks import SharedResNetwork
from common.state import State
from network_wrapper import NNWrapper
from common.game import Game
from common.nnmcts import NNMCTS
import copy
import numpy as np
import torch as T
from torch.nn.utils.clip_grad import clip_grad_norm_
from common.utils import get_device



class TrainerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self) -> None:
        pass


class PBTTrainer(TrainerBase):
    def __init__(self, game: Game, n_iterations: int, n_population: int, n_episodes: int, n_sims: int,n_testing_sets:int) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.n_sims = n_sims
        self.game = game
        self.n_population = n_population
        self.networks: List[TrainDroplet] = []
        self.n_testing_sets = n_testing_sets
        base_network = SharedResNetwork(self.game.observation_space,self.game.n_actions)
        self.initialize_droplets(n_population,base_network)

    def train(self) -> NNWrapper:
        strongest_network = copy.deepcopy(self.networks[0])
        examples : List[Tuple[State, np.ndarray, np.ndarray]]
        for i in range(self.n_iterations):
            assert self.n_episodes // (len(self.networks)*2) == 0
            n_rounds = int(self.n_episodes // (len(self.networks) * 2))
            examples = []
            for j in range(n_rounds):
                if j and j % 16 == 0:  # selfplay
                    for p in self.networks:
                        examples += self.execute_episode(p, p)
                else:  # play against each other
                    indices: np.ndarray
                    indices = np.arange(0, len(self.networks))
                    np.random.shuffle(indices)
                    n_matches = int(len(indices) // 2)
                    indices.reshape((2, n_matches))
                    for ind in indices:
                        p1 = self.networks[ind[0]]
                        p2 = self.networks[ind[1]]
                        examples += self.execute_episode(p1, p2)
                        examples += self.execute_episode(p2, p1)
            # Important TODO , Check if this line is correct
            states,probs,wdl = list(zip(*examples))
            obs = [state.to_obs() for state in states]

            # Training each network from examples
            for network in self.networks:
                if network:
                    self._train_network(network,obs,probs,wdl)
            
            # Evaluating networks
            players = [NNMCTSPlayer(self.game,network,self.n_sims) for network in self.networks]
            tournament = RoundRobin(self.game,players,self.n_testing_sets)
            results , rankings = tournament.start()
            networks = [n for n in self.networks]
            for j,network in enumerate(self.networks):
                network_rank = rankings[j]
                networks[network_rank] = network
            
            self.networks = networks
            n_replaced_networks = int(self.n_population // 5)
            i_first_replaced_network = self.n_population - n_replaced_networks
            for j in range(n_replaced_networks):
                top_network = self.networks[j]
                weak_network = self.networks[i_first_replaced_network+j]
                weak_network.perturb(top_network)
            
            # Evaluating top network vs strongest network so far
            top_network = self.networks[0]
            match = Match(self.game,top_network,strongest_network,n_sets=self.n_testing_sets*4)
            wdl = match.start()
            if wdl[0] > wdl[2]:
                strongest_network.nn.load_state_dict(top_network.nn.state_dict())
        
        return strongest_network

    def execute_episode(self, p1: NNWrapper, p2: NNWrapper):
        examples: List[Tuple[State, np.ndarray,
                             Union[np.ndarray, None], int]] = []
        game = copy.deepcopy(self.game)
        players = [NNMCTS(game, p1, self.n_sims),
                   NNMCTS(game, p2, self.n_sims)]
        state = game.reset()
        current_player = 0
        while True:
            player_mcts = players[current_player]
            probs = player_mcts.get_probs(state)
            examples.append((state, probs, None, current_player))
            action = np.random.choice(len(probs), p=probs)
            state = state.move(action)
            current_player = 1 - current_player
            if state.is_game_over():
                results: List[Tuple[State, np.ndarray, np.ndarray]] = self.assign_rewards(
                    examples, state.game_result(), current_player)
                return results

    def assign_rewards(self, examples, rewards, last_player) -> List[Tuple[State, np.ndarray, np.ndarray]]:
        inverted = rewards[::-1]
        for ex in examples:
            ex[2] = rewards if ex[3] == last_player else inverted
        return [(x[0], x[1], x[2]) for x in examples]
    
    def _train_network(self,network:TrainDroplet,obs:List[np.ndarray],probs:List[np.ndarray],wdl:List[np.ndarray]):
        optimzer = T.optim.Adam(network.nn.parameters(),network.lr,weight_decay=1e-4)
        obs_t = T.tensor(obs,dtype=T.float32,device=get_device())
        target_probs = T.tensor(probs,dtype=T.float32,device=get_device())
        target_wdl = T.tensor(wdl,dtype=T.float32,device=get_device())
        predicted_probs :Tensor
        predicted_wdl : Tensor
        predicted_probs ,predicted_wdl = network.nn(obs_t)
        actor_loss = self._loss(target_probs,predicted_probs)
        critic_loss = self._loss(target_wdl,predicted_wdl)
        total_loss = actor_loss + network.ratio * critic_loss
        optimzer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(network.nn.parameters(),0.5)
        optimzer.step()

    
    def _loss(self,target_probs:Tensor,predicted_probs:Tensor)->Tensor:
        log_probs = predicted_probs.log()
        loss = -(target_probs * log_probs).mean()
        return loss
    
    def initialize_droplets(self,n_population:int,base_network:NNBase):
        self.networks = []
        b_lr:np.ndarray = -3 * np.random.rand(n_population)
        learning_rates = (10 ** b_lr) * (10**-2)
        actor_critic__loss_ratios = np.random.rand(n_population) * 2
        for i in range(n_population):
            nnet = copy.deepcopy(base_network)
            droplet = TrainDroplet(nnet,lr=learning_rates[i],ratio=actor_critic__loss_ratios[i])
            self.networks.append(droplet)
            
