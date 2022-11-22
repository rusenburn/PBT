import torch
from common.networks.basic_networks import SharedResNetwork
from common.trainer import PBTTrainer,APbtTrainer
from games.othello.game import OthelloGame
from common.utils import get_device

def main():
    torch.set_num_threads(8)
    device = get_device()
    print(device)
    game_fn = lambda:OthelloGame()
    game = game_fn()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5)
    trainer = APbtTrainer(game_fn,20,16,256,50,10,8,2,network)
    
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','othello_nn.pt')
if __name__ == '__main__':
    main()