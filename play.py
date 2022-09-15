from time import time
from common.utils import get_device
from games.connect4.game import ConnectFourGame
from common.networks.basic_networks import SharedResNetwork
from common.arena.players import DualNNMCTSPlayer, NNMCTS2Player, NNMCTSPlayer,Human
from common.arena.match import Match 
from common.network_wrapper import TorchWrapper
from games.othello.game import OthelloGame
from games.tictactoe.game import TicTacToeGame

def main():
    device = get_device()
    print(device)
    game_fn = lambda: OthelloGame()
    game = game_fn()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    wrapper = TorchWrapper(network)
    network_2 = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    wrapper_2 = TorchWrapper(network_2)
    wrapper.load_check_point('tmp','othello_nn_20.pt')
    wrapper_2.load_check_point('tmp','othello_nn_20.pt')
    player_1 = NNMCTSPlayer(game.n_actions,wrapper,25)
    player_2 = NNMCTS2Player(game.n_actions,wrapper_2,25)
    # player_2 = Human()
    n_games= 26
    t_start = time()
    match = Match(game_fn,player_1,player_2,n_games,False)
    scores = match.start()
    duration = time()-t_start
    print(scores)
    print(f"time take\t {duration:0.2f} to play {n_games} for an average of {duration/n_games:0.2f} seconds per game")
if __name__ == '__main__':
    main()