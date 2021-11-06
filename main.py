from common.trainer import PBTTrainer
from games.tictactoe.game import TicTacToeGame
def main():
    game = TicTacToeGame()
    trainer = PBTTrainer(game,20,16,512,25,4,4,2)
    wrapper = trainer.train()
    wrapper.save_check_point('tmp','nn.pt')

if __name__ == '__main__':
    main()