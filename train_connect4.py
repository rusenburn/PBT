from common.networks.basic_networks import SharedResNetwork
from common.trainer import PBTTrainer
from games.connect4.game import ConnectFourGame
from common.utils import get_device
def main():
    device = get_device()
    print(device)
    game = ConnectFourGame()
    network = SharedResNetwork(game.observation_space,game.n_actions,n_blocks=5,)
    trainer = PBTTrainer(game,20,16,256,25,10,8,2,network)
    wrapper = trainer.train()
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','connect4nn.pt')
if __name__ == '__main__':
    main()