from common.networks.basic_networks import SharedResNetwork
from common.trainer import PBTTrainer, APbtTrainer
from games.connect4.game import ConnectFourGame
from common.utils import get_device


def main():
    device = get_device()
    print(device)
    def game_fn(): return ConnectFourGame()
    game = game_fn()
    network = SharedResNetwork(
        game.observation_space, game.n_actions, n_blocks=5)
    # trainer = PBTTrainer(game_fn,20,16,256,15,10,8,2,network)
    trainer = APbtTrainer(game_fn, 20, 16, 256, 50, 10, 8, 2, network,load_from_checkpoint=False)
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp', 'connect4nn.pt')


if __name__ == '__main__':
    main()
