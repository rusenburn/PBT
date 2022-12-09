from time import time
import torch as T
from common.evaluators import DeepNNEvaluator, RolloutPolicy,RandomRollouts
from common.utils import get_device
from games.connect4.game import ConnectFourGame
from common.networks.basic_networks import SharedResNetwork,RolloutPolicyNetwork,NoisySharedResNetwork,SharedRecResNetwork
from common.arena.players import  NNMCTS2Player, NNMCTSPlayer,Human,AMCTSPlayer,RandomActionPlayer
from common.arena.match import Match 
from common.network_wrapper import TorchWrapper
from games.othello.game import OthelloGame
from games.tictactoe.game import TicTacToeGame
from games.connect4.game import ConnectFourGame

def main():
    device = get_device()
    print(device)
    game_fn = lambda: ConnectFourGame()
    game = game_fn()

    network_1 = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    wrapper_1 = TorchWrapper(network_1)
    wrapper_1.load_check_point('tmp','connect4nn_60.pt')

    
    # player_1 = Human()
    # player_1 = NNMCTSPlayer(game.n_actions,wrapper_1,50,1)
    player_1 = NNMCTS2Player(game.n_actions,RandomRollouts(),500,temperature=0.1)
    # player_1 = NNMCTS2Player(game.n_actions,DeepNNEvaluator(wrapper_1),50,temperature=1)
    # player_1 = RandomActionPlayer()
    # player_1 = AMCTSPlayer(game.n_actions,wrapper_1,50,0.1)


    network_2 = SharedRecResNetwork(game.observation_space,game.n_actions,n_blocks=8)
    wrapper_2 = TorchWrapper(network_2)
    wrapper_2.load_check_point('tmp','checkpoint_nnet_strongest.pt')
    # player_2 =  Human()
    # player_2 =  NNMCTS2Player(game.n_actions,DeepNNEvaluator(wrapper_2),50,temperature=1)
    # player_2 = NNMCTSPlayer(game.n_actions,wrapper_2,50,1)
    player_2 = AMCTSPlayer(game.n_actions,wrapper_2,200,temperature=0.1)
    n_games = 10
    t_start = time()
    match = Match(game_fn,player_1,player_2,n_games,True)
    scores = match.start()
    duration = time()-t_start
    print(scores)
    print(f"time take\t {duration:0.2f} to play {n_games} for an average of {duration/n_games:0.2f} seconds per game")


    
if __name__ == '__main__':
    main()