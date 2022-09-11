import torch
from common.networks.basic_networks import SharedResNetwork
from common.trainer import PBTTrainer
from games.othello.game import OthelloGame
from common.utils import get_device
def main():
    device = get_device()
    print(device)
    game = OthelloGame()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    trainer = PBTTrainer(game,20,16,256,15,10,8,2,network)
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','othello_nn.pt')
if __name__ == '__main__':
    main()