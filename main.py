from common.trainer import PBTTrainer
from games.tictactoe.game import TicTacToeGame
from common.utils import get_device
def main():
    device = get_device()
    print(device)
    game = TicTacToeGame()
    trainer = PBTTrainer(game,20,16,256,25,10,8,2)
    wrapper = trainer.train()
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','nn.pt')
if __name__ == '__main__':
    main()