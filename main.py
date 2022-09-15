import torch
from common.trainer import PBTTrainer
from games.tictactoe.game import TicTacToeGame
from common.utils import get_device
def main():
    torch.set_num_threads(8)
    device = get_device()
    print(device)
    game_fn = lambda:TicTacToeGame()
    trainer = PBTTrainer(game_fn,20,16,16,25,10,8,2)
    wrapper = trainer.train()
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','nn.pt')
if __name__ == '__main__':
    main()