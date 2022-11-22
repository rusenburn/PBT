import torch
from common.networks.basic_networks import SharedResNetwork
from common.trainer import PBTTrainer
from games.othello.game import OthelloGame
from common.utils import get_device
from trainers.rollout_policies_trainer import RolloutPoliciesTrainer


'''
WARNING : Tooooo slow Not optimized yet
'''
def main():
    torch.set_num_threads(8)
    device = get_device()
    print(device)
    game_fn = lambda:OthelloGame()
    game = game_fn()
    trainer = RolloutPoliciesTrainer(game_fn,20,16,16,25,10,8,2)
    for wrapper in trainer.train():
        wrapper.save_check_point('tmp','othello_nn_rollout.pt')
if __name__ == '__main__':
    main()