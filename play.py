from common.utils import get_device
from games.connect4.game import ConnectFourGame
from common.networks.basic_networks import SharedResNetwork
from common.arena.players import NNMCTSPlayer,Human
from common.arena.match import Match 
from common.network_wrapper import TorchWrapper
from games.tictactoe.game import TicTacToeGame

def main():
    device = get_device()
    print(device)
    game = ConnectFourGame()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    wrapper = TorchWrapper(network)
    network_2 = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    wrapper_2 = TorchWrapper(network_2)
    wrapper.load_check_point('tmp','connect4nn.pt')
    wrapper_2.load_check_point('tmp','connect4nn_0.pt')
    player_1 = NNMCTSPlayer(game,wrapper,100)
    player_2 = Human()
    match = Match(game,player_1,player_2,10,True)
    scores = match.start()
    print(scores)
if __name__ == '__main__':
    main()