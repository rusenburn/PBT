import copy
import time
import os
import concurrent.futures

import numpy as np
import torch as T

from typing import Callable, Iterator
from torch.nn.utils.clip_grad import clip_grad_norm_

from common.amcts import AMCTS
from common.arena.match import Match
from common.arena.players import AMCTSPlayer, NNMCTS2Player, PlayerBase
from common.game import Game
from common.networks.basic_networks import SharedResNetwork
from common.nnmcts import NNMCTS2, MctsBase
from common.evaluators import DeepNNEvaluator
from common.state import State
from common.trainer import TrainerBase
from common.networks.base import NNBase
from common.network_wrapper import NNWrapper, TorchWrapper
from common.utils import get_device


class AlphaZeroTrainerBase(TrainerBase):
    def __init__(self,
                 game_fn: Callable[[], Game], n_iterations: int,
                 n_episodes: int, n_sims: int, n_epochs: int,
                 n_batches, lr: float, actor_critic_ratio: float, n_testing_episodes: int,
                 network: NNBase,
                 use_async_mcts=False, checkpoint: str | None = None) -> None:

        super().__init__()
        self.game_fn = game_fn
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.n_sims = n_sims
        self.lr = lr
        self.actor_critic_ratio = actor_critic_ratio
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.n_testing_episodes = n_testing_episodes
        self.use_async_mcts = use_async_mcts
        self.checkpoint: str | None = checkpoint
        game = self.game_fn()
        self.n_game_actions = game.n_actions
        base_network = SharedResNetwork(
            game.observation_space, game.n_actions, n_blocks=5) if network is None else network
        self.network_wrapper: NNWrapper = self._initialize_wrappers(
            base_network, checkpoint)

    def train(self) -> Iterator[NNWrapper]:
        strongest_wrapper = copy.deepcopy(self.network_wrapper)
        examples: list[tuple[State, np.ndarray, np.ndarray]]
        for iteration in range(self.n_iterations):
            t_collecting_start = time.perf_counter()
            print(f"Iteration {iteration+1} out of {self.n_iterations}")
            n_workers = 8
            examples = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                a = executor.map(self.execute_episode, [
                                 self.network_wrapper for _ in range(self.n_episodes)])
                examples += [y for x in a for y in x]

            states, probs, wdl = list(zip(*examples))
            # states = [ex[0] for ex in examples]
            # probs = [ex[1] for ex in examples]
            # wdl = [ex[2] for ex in examples]
            obs = [state.to_obs() for state in states]
            collecting_data_duration = time.perf_counter() - t_collecting_start
            t_training_start = time.perf_counter()
            n_examples: int = len(examples)
            print(f"Training Phase using {n_examples} examples...")
            self._train_network(self.network_wrapper, obs, probs, wdl)
            training_duration = time.perf_counter()-t_training_start
            t_evaluation_start = time.perf_counter()
            if iteration % 2 == 1 or iteration == 0:
                print("Evaluation phase...")
                temperature = 0.5
                current_network_player: PlayerBase
                strongest_network_player: PlayerBase
                if self.use_async_mcts:
                    current_network_player = AMCTSPlayer(
                        self.n_game_actions, self.network_wrapper, self.n_sims, temperature=temperature)
                    strongest_network_player = AMCTSPlayer(
                        self.n_game_actions, strongest_wrapper, self.n_sims, temperature=temperature)
                else:
                    current_evaluator = DeepNNEvaluator(self.network_wrapper)
                    current_network_player = NNMCTS2Player(
                        self.n_game_actions, current_evaluator, self.n_sims, temperature=temperature)
                    strongest_evaluator = DeepNNEvaluator(strongest_wrapper)
                    strongest_network_player = NNMCTS2Player(
                        self.n_game_actions, strongest_evaluator, self.n_sims, temperature=temperature)

                match_ = Match(self.game_fn, current_network_player,
                               strongest_network_player, self.n_testing_episodes, render=False)
                wdl = match_.start()
                # score ratio = ( win * 2 + draw ) / ( total played * 2 )
                score_ratio = (wdl[0]*2 + wdl[1]) / (wdl.sum()*2)
                print(f"wins : {wdl[0]} , draws:{wdl[1]} , losses:{wdl[2]}")

                print(
                    f"score ratio against old strongest opponent: {score_ratio*100:0.2f}%")
                if wdl[0] > wdl[2]:
                    strongest_wrapper.nn.load_state_dict(
                        self.network_wrapper.nn.state_dict())
            evaluation_duration = time.perf_counter() - t_evaluation_start
            iteration_duration = time.perf_counter() - t_collecting_start
            print("**************************************************************")
            print(
                f"Iteration                  {iteration+1} of {self.n_iterations}")
            print(
                f"Iteration Duration         {iteration_duration:0.2f} seconds")
            print(
                f"Collecting Data Duration   {collecting_data_duration:0.2f} seconds")
            print(
                f"Training Duration          {training_duration:0.2f} seconds")
            print(
                f"Evaluation Duration        {evaluation_duration:0.2f} seconds")
            print(
                f"Training Data Count        {n_examples} examples")
            print("**************************************************************")
            # TODO set checkpoint

            yield strongest_wrapper
        return strongest_wrapper

    def _train_network(self, network_wrapper: NNWrapper, obs: list[np.ndarray], probs: list[np.ndarray], wdl: list[np.ndarray]):
        device = get_device()
        network_wrapper.nn.train()
        optimzer = T.optim.Adam(
            network_wrapper.nn.parameters(), self.lr, weight_decay=1e-4)

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

                predicted_probs: T.Tensor
                predicted_wdl: T.Tensor
                predicted_probs, predicted_wdl = network_wrapper.nn(obs_t)
                actor_loss = self.cross_entropy_loss(
                    target_probs, predicted_probs)
                critic_loss = self.cross_entropy_loss(
                    target_wdl, predicted_wdl)
                total_loss = actor_loss + self.actor_critic_ratio * critic_loss
                optimzer.zero_grad()
                total_loss.backward()
                clip_grad_norm_(network_wrapper.nn.parameters(), 0.5)
                optimzer.step()
        if T.cuda.is_available():
            T.cuda.empty_cache()

    @staticmethod
    def cross_entropy_loss(target_probs: T.Tensor, predicted_probs: T.Tensor):
        log_probs = predicted_probs.log()
        loss = -(target_probs*log_probs).sum(dim=-1).mean()
        return loss

    def execute_episode(self, wrapper: NNWrapper):
        examples: list[tuple[State, np.ndarray, np.ndarray | None, int]] = []
        results: list[tuple[State, np.ndarray, np.ndarray]] = []
        game = self.game_fn()
        temperature = 1.0
        cpuct = 2.0
        player: MctsBase
        if self.use_async_mcts:
            player = AMCTS(self.n_game_actions, wrapper.nn,
                           self.n_sims, 0, cpuct, temperature)
        else:
            evaluator = DeepNNEvaluator(wrapper)
            player = NNMCTS2(self.n_game_actions, evaluator,
                             self.n_sims, cpuct, temperature)
        state = game.reset()
        current_player = 0
        while True:
            probs = player.search(state)
            examples.append((state, probs, None, current_player))
            syms = state.get_symmetries(probs)
            for s, p in syms:
                examples.append((s, p, None, current_player))
            action = np.random.choice(len(probs), p=probs)
            state = state.move(action)
            current_player = 1-current_player
            if state.is_game_over():
                results = self._assign_rewards(
                    examples, state.game_result(), current_player)
                break
        return results

    def _assign_rewards(self, examples: list[tuple[State, np.ndarray, np.ndarray | None, int]], rewards: np.ndarray, last_player: int):
        inverted: np.ndarray = rewards[::-1]
        results = [(ex[0], ex[1], rewards if ex[3] ==
                    last_player else inverted) for ex in examples]
        return results

    def _initialize_wrappers(self, network: NNBase, checkpoint: str | None):
        if checkpoint:
            nnet = copy.deepcopy(network)
            if os.path.exists(checkpoint):
                nnet.load_model(checkpoint)
                wrapper = TorchWrapper(nnet)
                return wrapper

        print("creating a network...")
        nnet = copy.deepcopy(network)
        wrapper = TorchWrapper(nnet)
        return wrapper


class AsyncAlpaZeoTrainer(AlphaZeroTrainerBase):
    def __init__(self, game_fn: Callable[[], Game], n_iterations: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches, lr: float, actor_critic_ratio: float, n_testing_episodes: int, network: NNBase, checkpoint: str | None = None) -> None:
        super().__init__(game_fn, n_iterations, n_episodes, n_sims, n_epochs, n_batches,
                         lr, actor_critic_ratio, n_testing_episodes, network, True, checkpoint)


class AlpaZeroTrainer(AlphaZeroTrainerBase):
    def __init__(self, game_fn: Callable[[], Game], n_iterations: int, n_episodes: int, n_sims: int, n_epochs: int, n_batches, lr: float, actor_critic_ratio: float, n_testing_episodes: int, network: NNBase, checkpoint: str | None = None) -> None:
        super().__init__(game_fn, n_iterations, n_episodes, n_sims, n_epochs, n_batches,
                         lr, actor_critic_ratio, n_testing_episodes, network, False, checkpoint)
