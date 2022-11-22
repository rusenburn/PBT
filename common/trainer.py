from abc import ABC, abstractmethod
import copy
import time
import os
import torch as T
import numpy as np
import concurrent.futures
import json
from tqdm import tqdm, trange
from torch.nn.utils.clip_grad import clip_grad_norm_
from typing import Callable, Iterator, List, Tuple, Union
from torch.functional import Tensor


from common.evaluators import DeepNNEvaluator
from common.arena.match import Match
from common.arena.players import NNMCTSPlayer, PlayerBase, AMCTSPlayer
from common.arena.roundrobin import RoundRobin
from common.network_wrapper import TorchWrapper, TrainDroplet, NNWrapper
from common.networks.base import NNBase
from common.networks.basic_networks import SharedResNetwork
from common.state import State
from common.game import Game
from common.nnmcts import NNMCTS, NNMCTS2, MctsBase
from common.amcts import AMCTS
from common.utils import get_device , json_load ,json_dump


class TrainerBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self) -> Iterator[NNWrapper]:
        pass


class PBTTrainerBase(TrainerBase):
    def __init__(self, game_fn: Callable[[], Game], n_iterations: int, n_population: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches: int, n_testing_sets: int, network: SharedResNetwork | None = None, use_async_mcts=False,load_from_checkpoint=False) -> None:
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
        self.use_async_mcts = use_async_mcts
        self.load_from_checkpoint = load_from_checkpoint
        game = self.game_fn()

        base_network = SharedResNetwork(
            game.observation_space, game.n_actions) if network is None else network
        self.networks = self.initialize_droplets(n_population, base_network)
        self.n_game_actions = game.n_actions

    def train(self) -> Iterator[NNWrapper]:
        strongest_network = copy.deepcopy(self.networks[0])
        examples: List[Tuple[State, np.ndarray, np.ndarray]]
        for iteration in range(self.n_iterations):

            t_collecting_start = time.time()
            assert self.n_episodes % (len(self.networks)) == 0
            print(f'Iteration {iteration+1} out of {self.n_iterations}')
            n_rounds = int(self.n_episodes // (len(self.networks)))
            examples = []

            for round_num in trange(n_rounds, desc="Collecting Data"):

                executes = []

                if round_num % 8 == 0:  # selfplay
                    for p in self.networks:
                        executes.append((p, p))

                else:  # play against each other
                    indices: np.ndarray
                    indices = np.arange(0, len(self.networks))
                    np.random.shuffle(indices)
                    n_matches = int(len(indices) // 2)
                    indices = indices.reshape((n_matches, 2))
                    for ind in indices:
                        p1 = self.networks[ind[0]]
                        p2 = self.networks[ind[1]]
                        executes.append((p1, p2))
                        executes.append((p2, p1))
                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    a = executor.map(self.execute_episode_process, executes)
                    examples += [y for x in a for y in x]

            states, probs, wdl = list(zip(*examples))
            obs = [state.to_obs() for state in states]
            t_training_start = time.time()

            # Training each network from examples
            print(f'Training Using {len(examples)} Examples...')
            for round_num, network in enumerate(tqdm(self.networks, desc="Training Networks")):
                if network:
                    self._train_network(network, obs, probs, wdl)

            # Evaluating networks
            t_evalutuaion_1_start = time.time()
            if iteration % 2 == 1:
                print('Evaluation Phase...')
                temperature = 0.5
                players: List[PlayerBase]
                if (self.use_async_mcts):
                    players = [AMCTSPlayer(
                        self.n_game_actions, network, self.n_sims, temperature)for network in self.networks]

                else:
                    players = [NNMCTSPlayer(self.n_game_actions, network, self.n_sims, temperature)
                               for network in self.networks]
                tournament = RoundRobin(
                    self.game_fn, players, self.n_testing_sets)
                results, rankings = tournament.start(print_progress=True)
                print(rankings)
                networks = [n for n in self.networks]
                for round_num, network in enumerate(self.networks):
                    network_rank = rankings[round_num]
                    networks[network_rank] = network

                self.networks = networks
                n_replaced_networks = int(self.n_population // 5)
                for round_num in range(n_replaced_networks):
                    top_network = self.networks[round_num]
                    weak_network = self.networks[-round_num-1]
                    old_lr = weak_network.lr
                    old_r = weak_network.ratio
                    weak_network.perturb(top_network)
                    print(
                        f'Changing network hyperparameters\nlr :{old_lr:0.2e} -> {weak_network.lr:0.2e}\nratio:{old_r:0.2e} -> {weak_network.ratio:0.2e}\n')

                # Evaluating top network vs strongest network so far
                t_evalutuaion_2_start = time.time()
                top_network = self.networks[0]
                top_network_player: PlayerBase = AMCTSPlayer(self.n_game_actions, top_network, self.n_sims, temperature=temperature) if self.use_async_mcts else NNMCTSPlayer(
                    self.n_game_actions, top_network, self.n_sims, temperature=temperature)

                strongest_network_player = AMCTSPlayer(self.n_game_actions, strongest_network, self.n_sims, temperature=temperature) if self.use_async_mcts else NNMCTSPlayer(
                    self.n_game_actions, strongest_network, self.n_sims, temperature=temperature)

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

            print(f"Iteration\t\t {iteration+1}")
            print(f"Iteration Duration\t\t {iteration_duration:0.2f}")
            print(
                f"Collecting Data Duration\t\t {collecting_data_duration:0.2f}")
            print(f"Training Duration\t\t {training_duration:0.2f}")
            if iteration % 2 == 1:
                evaluation_1_duration = t_evalutuaion_2_start - t_evalutuaion_1_start
                evaluation_2_duration = t_iteration_end - t_evalutuaion_2_start
                print(f"Evaluation Phase1\t\t {evaluation_1_duration:0.2f}")
                print(f"Evaluation Phase2\t\t {evaluation_2_duration:0.2f}")
            

            ##### Setting Checkpoint #####
            dictionary : dict = {}
            strongest_network_path = os.path.join("tmp",f"checkpoint_nnet_strongest.pt")
            strongest_network.save_check_point("tmp",f"checkpoint_nnet_strongest.pt")
            dictionary["strongest"] = {"base_lr":strongest_network.base_lr,"lr":strongest_network.lr,"ratio":strongest_network.ratio,"file_path":strongest_network_path}
            for _id,drop in enumerate(self.networks):
                file_path = os.path.join("tmp",f"checkpoint_nnet_{_id}.pt")
                drop.save_check_point("tmp",f"checkpoint_nnet_{_id}.pt")
                dictionary[_id] = {"base_lr":drop.base_lr,"lr":drop.lr,"ratio":drop.ratio,"file_path":file_path}
            json_dump(dictionary)
            
            yield strongest_network
        return strongest_network

    def execute_episode_process(self, args):
        p1: NNWrapper = args[0]
        p2: NNWrapper = args[1]
        return self.execute_episode(p1, p2)

    def execute_episode(self, p1: NNWrapper, p2: NNWrapper):
        examples: List[Tuple[State, np.ndarray,
                             Union[np.ndarray, None], int]] = []
        game = self.game_fn()

        players: list[MctsBase]
        temperature = 1.0
        cpuct = 2.0
        if self.use_async_mcts:
            
            players = [AMCTS(self.n_game_actions, p1.nn, self.n_sims, duration_in_millis=0, c=cpuct, temperature=temperature),
                       AMCTS(self.n_game_actions, p2.nn, self.n_sims,
                             duration_in_millis=0, c=cpuct, temperature=temperature)
                       ]
        else:
            players = [NNMCTS(self.n_game_actions, p1, self.n_sims,cpuct=cpuct,temperature=temperature),
                       NNMCTS(self.n_game_actions, p2, self.n_sims,cpuct=cpuct,temperature=temperature)]
        state = game.reset()
        current_player = 0
        while True:
            player_mcts = players[current_player]
            probs = player_mcts.search(state)
            examples.append((state, probs, None, current_player))
            syms = state.get_symmetries(probs)
            for s, p in syms:
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
        optimzer = T.optim.Adam(network.nn.parameters(),
                                network.lr, weight_decay=1e-4)
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

    def initialize_droplets(self, n_population: int, base_network: NNBase) -> List[TrainDroplet]:
        self.networks = []
        if self.load_from_checkpoint:
            try :
                settings :dict|None= json_load()
                if settings is not None and len(settings.keys()) >= n_population:
                    for i in range(n_population):
                        droplet_dict = settings[str(i)]
                        nnet = copy.deepcopy(base_network)
                        nnet.load_model(droplet_dict["file_path"])
                        droplet = TrainDroplet(nnet,lr=droplet_dict["lr"],ratio=droplet_dict["ratio"])
                        droplet.base_lr = droplet_dict["base_lr"]
                        self.networks.append(droplet)
                        print(f"network {i} has been added successfully.")
                    return self.networks
            except Exception as e:
                print(e)

        print("Generating new networks...")
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


class PBTTrainer(PBTTrainerBase):
    def __init__(self, game_fn: Callable[[], Game], n_iterations: int, n_population: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches: int, n_testing_sets: int, network: SharedResNetwork | None = None,load_from_checkpoint=False) -> None:
        super().__init__(game_fn, n_iterations, n_population, n_episodes, n_sims,
                         n_epochs, n_batches, n_testing_sets, network, use_async_mcts=False,load_from_checkpoint=load_from_checkpoint)


class APbtTrainer(PBTTrainerBase):
    def __init__(self, game_fn: Callable[[], Game], n_iterations: int, n_population: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches: int, n_testing_sets: int, network: SharedResNetwork | None = None,load_from_checkpoint=False) -> None:
        super().__init__(game_fn, n_iterations, n_population, n_episodes, n_sims,
                         n_epochs, n_batches, n_testing_sets, network, use_async_mcts=True,load_from_checkpoint=load_from_checkpoint)
