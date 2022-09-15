from abc import ABC, abstractmethod
import time
import concurrent.futures
from typing import Callable, Iterator, List, Tuple, Union

from common.arena.match import Match
from torch.functional import Tensor
from common.arena.players import NNMCTSPlayer
from common.arena.roundrobin import RoundRobin
from common.network_wrapper import TorchWrapper, TrainDroplet, NNWrapper
from common.networks.base import NNBase
from common.networks.basic_networks import SharedResNetwork
from common.state import State
from common.game import Game
from common.nnmcts import NNMCTS
import copy
from tqdm import tqdm,trange
import numpy as np
import torch as T
from torch.nn.utils.clip_grad import clip_grad_norm_
from common.utils import get_device

class TrainerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self) -> Iterator[NNWrapper]:
        pass


class PBTTrainer(TrainerBase):
    def __init__(self, game_fn: Callable[[],Game], n_iterations: int, n_population: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches: int, n_testing_sets: int,network:SharedResNetwork|None=None) -> None:
        super().__init__()
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.n_sims = n_sims
        self.game_fn = game_fn
        self.n_population = n_population
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.networks: List[TrainDroplet] = []
        self.n_testing_sets = n_testing_sets
        game = self.game_fn()
        base_network = SharedResNetwork(
            game.observation_space, game.n_actions) if network is None else network
        self.networks = self.initialize_droplets(n_population, base_network)
        self.n_game_actions = game.n_actions

    def train(self) -> Iterator[NNWrapper]:
        strongest_network = copy.deepcopy(self.networks[0])
        examples: List[Tuple[State, np.ndarray, np.ndarray]]
        for i in range(self.n_iterations):
            t_collecting_start = time.time()
            assert self.n_episodes % (len(self.networks)) == 0
            print(f'Iteration {i+1} out of {self.n_iterations}')
            n_rounds = int(self.n_episodes // (len(self.networks)))
            examples = []
            for j in trange(n_rounds,desc="Collecting Data"):
                executes = []
                if j % 8 == 0:  # selfplay
                    for p in self.networks:
                        # examples += self.execute_episode(p, p)
                        executes.append((p,p))
                else:  # play against each other
                    indices: np.ndarray
                    indices = np.arange(0, len(self.networks))
                    np.random.shuffle(indices)
                    n_matches = int(len(indices) // 2)
                    indices = indices.reshape((n_matches, 2))
                    for ind in indices:
                        p1 = self.networks[ind[0]]
                        p2 = self.networks[ind[1]]
                        executes.append((p1,p2))
                        executes.append((p2,p1))
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    a = executor.map(self.execute_episode_process, executes)
                    examples+= [y for x in a for y in x]

            states, probs, wdl = list(zip(*examples))
            obs = [state.to_obs() for state in states]
            t_training_start = time.time()
            # Training each network from examples
            print(f'Training Using {len(examples)} Examples...')
            for j, network in enumerate(tqdm(self.networks,desc="Training Networks")):
                if network:
                    # print(f'Training Network No {j}.')
                    self._train_network(network, obs, probs, wdl)

            # Evaluating networks
            t_evalutuaion_1_start = time.time()
            if i % 2 ==1 :
                print('Evaluation Phase...')
                players: List[NNMCTSPlayer] = [NNMCTSPlayer(self.n_game_actions, network, self.n_sims)
                                            for network in self.networks]
                tournament = RoundRobin(self.game_fn, players, self.n_testing_sets)
                results, rankings = tournament.start(print_progress=True)
                print(rankings)
                networks = [n for n in self.networks]
                for j, network in enumerate(self.networks):
                    network_rank = rankings[j]
                    networks[network_rank] = network

                self.networks = networks
                n_replaced_networks = int(self.n_population // 5)
                for j in range(n_replaced_networks):
                    top_network = self.networks[j]
                    weak_network = self.networks[-j-1]
                    old_lr = weak_network.lr
                    old_r = weak_network.ratio
                    weak_network.perturb(top_network)
                    print(
                        f'Changing network hyperparameters\nlr :{old_lr:0.2e} -> {weak_network.lr:0.2e}\nratio:{old_r:0.2e} -> {weak_network.ratio:0.2e}\n')

                # Evaluating top network vs strongest network so far
                t_evalutuaion_2_start = time.time()
                top_network = self.networks[0]
                top_network_player = NNMCTSPlayer(
                    self.n_game_actions, top_network, self.n_sims)
                strongest_network_player = NNMCTSPlayer(
                    self.n_game_actions, strongest_network, self.n_sims)

                match = Match(self.game_fn, top_network_player,
                            strongest_network_player, n_sets=self.n_testing_sets*self.n_population)
                wdl = match.start()
                win_ratio = (wdl[0]*2 + wdl[1])/(wdl.sum() * 2)
                print(wdl)
                print(
                    f'win ratio against old strongest opponent: {win_ratio*100:0.2f}%')
                if wdl[0] > wdl[2]:
                    strongest_network.nn.load_state_dict(
                        top_network.nn.state_dict())
            t_iteration_end = time.time()
            iteration_duration = t_iteration_end - t_collecting_start
            collecting_data_duration = t_training_start - t_collecting_start
            training_duration = t_evalutuaion_1_start - t_training_start
            
            print(f"Iteration\t\t {i+1}")
            print(f"Iteration Duration\t\t {iteration_duration:0.2f}")
            print(f"Collecting Data Duration\t\t {collecting_data_duration:0.2f}")
            print(f"Training Duration\t\t {training_duration:0.2f}")
            if i%2 == 1:
                evaluation_1_duration = t_evalutuaion_2_start - t_evalutuaion_1_start
                evaluation_2_duration = t_iteration_end - t_evalutuaion_2_start
                print(f"Evaluation Phase1\t\t {evaluation_1_duration:0.2f}")
                print(f"Evaluation Phase2\t\t {evaluation_2_duration:0.2f}")
            yield strongest_network
            # self.n_sims+=1
        return strongest_network

    def execute_episode_process(self,args):
        p1 : NNWrapper = args[0]
        p2 : NNWrapper = args[1]
        return self.execute_episode(p1,p2)

    def execute_episode(self, p1: NNWrapper, p2: NNWrapper):
        examples: List[Tuple[State, np.ndarray,
                             Union[np.ndarray, None], int]] = []
        game = self.game_fn()
        players = [NNMCTS(self.n_game_actions, p1, self.n_sims),
                   NNMCTS(self.n_game_actions, p2, self.n_sims)]
        state = game.reset()
        current_player = 0
        while True:
            player_mcts = players[current_player]
            probs = player_mcts.get_probs(state)
            examples.append((state,probs,None,current_player))
            syms = state.get_symmetries(probs)
            for s,p in syms:
                examples.append((s, p, None, current_player))
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
        fixed_examples = [(ex[0], ex[1], rewards if ex[3] ==
                                   last_player else inverted) for ex in examples]
        return fixed_examples

    def _train_network(self, network: TrainDroplet, obs: List[np.ndarray], probs: List[np.ndarray], wdl: List[np.ndarray]):
        device = get_device()
        network.nn.train()
        optimzer = T.optim.Adam(network.nn.parameters(), network.lr,weight_decay=1e-4)
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
                target_probs = T.tensor(
                    probs_batch, dtype=T.float32, device=device)
                target_wdl = T.tensor(
                    wdl_batch, dtype=T.float32, device=device)
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
                
        if T.cuda.is_available():
            T.cuda.empty_cache()

    def _loss(self, target_probs: Tensor, predicted_probs: Tensor) -> Tensor:
        log_probs = predicted_probs.log()
        loss = -(target_probs * log_probs).sum(dim=-1).mean()
        # a1 = ((targets - outputs)**2).sum(dim=-1).mean()
        return loss

    def initialize_droplets(self, n_population: int, base_network: NNBase)->List[TrainDroplet]:
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
        return self.networks
