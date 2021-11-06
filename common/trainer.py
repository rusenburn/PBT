from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from common.arena.match import Match
from torch.functional import Tensor
from common.arena.players import NNMCTSPlayer
from common.arena.roundrobin import RoundRobin
from common.network_wrapper import TrainDroplet, NNWrapper
from common.networks.base import NNBase
from common.networks.basic_networks import SharedResNetwork
from common.state import State
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
    def train(self) -> NNWrapper:
        pass


class PBTTrainer(TrainerBase):
    def __init__(self, game: Game, n_iterations: int, n_population: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches: int, n_testing_sets: int) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.n_sims = n_sims
        self.game = game
        self.n_population = n_population
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.networks: List[TrainDroplet] = []
        self.n_testing_sets = n_testing_sets
        base_network = SharedResNetwork(
            self.game.observation_space, self.game.n_actions)
        self.initialize_droplets(n_population, base_network)

    def train(self) -> NNWrapper:
        strongest_network = copy.deepcopy(self.networks[0])
        examples: List[Tuple[State, np.ndarray, np.ndarray]]
        for i in range(self.n_iterations):
            assert self.n_episodes % (len(self.networks)) == 0
            print(f'Iteration {i+1} out of {self.n_iterations}')
            print('Collecting Data')
            n_rounds = int(self.n_episodes // (len(self.networks)))
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
                    indices = indices.reshape((n_matches, 2))
                    for ind in indices:
                        p1 = self.networks[ind[0]]
                        p2 = self.networks[ind[1]]
                        examples += self.execute_episode(p1, p2)
                        examples += self.execute_episode(p2, p1)
            # Important TODO , Check if this line is correct
            states, probs, wdl = list(zip(*examples))
            obs = [state.to_obs() for state in states]

            # Training each network from examples
            print(f'Training Using {len(examples)} Examples.')
            for j, network in enumerate(self.networks):
                if network:
                    print(f'Training Network No {j}.')
                    self._train_network(network, obs, probs, wdl)

            # Evaluating networks
            print('Evaluation Phase...')
            players: List[NNMCTSPlayer] = [NNMCTSPlayer(self.game, network, self.n_sims)
                                           for network in self.networks]
            tournament = RoundRobin(self.game, players, self.n_testing_sets)
            results, rankings = tournament.start()
            print(rankings)
            networks = [n for n in self.networks]
            for j, network in enumerate(self.networks):
                network_rank = rankings[j]
                networks[network_rank] = network

            self.networks = networks
            n_replaced_networks = int(self.n_population // 5)
            i_first_replaced_network = self.n_population - n_replaced_networks
            for j in range(n_replaced_networks):
                top_network = self.networks[j]
                weak_network = self.networks[-j-1]
                old_lr = weak_network.lr
                old_r = weak_network.ratio
                weak_network.perturb(top_network)
                print(f'changing a network hyperparameters\nlr :{old_lr:0.2e} -> {weak_network.lr:0.2e}\nratio:{old_r:0.2e} -> {weak_network.ratio:0.2e}\n')

            # Evaluating top network vs strongest network so far
            top_network = self.networks[0]
            top_network_player = NNMCTSPlayer(
                self.game, top_network, self.n_sims)
            strongest_network_player = NNMCTSPlayer(
                self.game, strongest_network, self.n_sims)
            match = Match(self.game, top_network_player,
                          strongest_network_player, n_sets=self.n_testing_sets*self.n_population)
            wdl = match.start()
            print(wdl)
            if wdl[0] > wdl[2]:
                strongest_network.nn.load_state_dict(
                    top_network.nn.state_dict())

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

    def assign_rewards(self,
                       examples: List[Tuple[State, np.ndarray,
                                            Union[np.ndarray, None], int]],
                       rewards: np.ndarray,
                       last_player) -> List[Tuple[State, np.ndarray, np.ndarray]]:

        inverted: np.ndarray = rewards[::-1]
        fixed_examples: List[Tuple[State, np.ndarray, np.ndarray]] = []
        for ex in examples:
            fixed_examples.append((ex[0], ex[1], rewards if ex[3] ==
                                   last_player else inverted))
            # ex[2] = rewards if ex[3] == last_player else inverted
        return fixed_examples

    def _train_network(self, network: TrainDroplet, obs: List[np.ndarray], probs: List[np.ndarray], wdl: List[np.ndarray]):
        device = get_device()
        network.nn.train()
        optimzer = T.optim.Adam(network.nn.parameters(), network.lr)
        obs_ar = np.array(obs)
        probs_ar = np.array(probs)
        wdl_ar = np.array(wdl)
        batch_size = int(len(obs)//self.n_batches)
        for epoch in range(self.n_epochs):
            sample_ids: np.ndarray
            for _ in range(self.n_batches):
                sample_ids = np.random.randint(len(obs), size=batch_size)
                obs_batch: np.ndarray = obs_ar[sample_ids]
                probs_batch: np.ndarray = probs_ar[sample_ids]
                wdl_batch: np.ndarray = wdl_ar[sample_ids]
                obs_t = T.tensor(obs_batch, dtype=T.float32, device=device)
                target_probs = T.tensor(probs_batch, dtype=T.float32, device=device)
                target_wdl = T.tensor(wdl_batch, dtype=T.float32, device=device)
                predicted_probs: Tensor
                predicted_wdl: Tensor
                predicted_probs, predicted_wdl = network.nn(obs_t)
                actor_loss = self._loss(target_probs, predicted_probs)
                critic_loss = self._loss(target_wdl, predicted_wdl)
                total_loss = actor_loss + network.ratio * critic_loss
                optimzer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(network.nn.parameters(), 0.5)
                optimzer.step()

        # obs_t = T.tensor(np.array(obs), dtype=T.float32, device=get_device())
        # target_probs = T.tensor(np.array(probs), dtype=T.float32, device=get_device())
        # target_wdl = T.tensor(np.array(wdl), dtype=T.float32, device=get_device())

    def _loss(self, target_probs: Tensor, predicted_probs: Tensor) -> Tensor:
        log_probs = predicted_probs.log()
        loss = -(target_probs * log_probs).mean()
        # a1 = ((targets - outputs)**2).sum(dim=-1).mean()
        return loss

    def initialize_droplets(self, n_population: int, base_network: NNBase):
        self.networks = []

        # Get Exponential Distribution Learning Rate
        b_lr: np.ndarray = -3 * np.random.rand(n_population)
        learning_rates = (10 ** b_lr) * (10**-2)

        # Get uniform disribution critic to actor loss ratio
        actor_critic__loss_ratios = np.random.rand(n_population) * 2
        for i in range(n_population):
            nnet = copy.deepcopy(base_network)
            droplet = TrainDroplet(
                nnet, lr=learning_rates[i], ratio=actor_critic__loss_ratios[i])
            self.networks.append(droplet)
