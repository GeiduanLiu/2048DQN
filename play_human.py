import argparse
import environment.game_2048

parser = argparse.ArgumentParser(description='2048 DQN model')
parser.add_argument('--reward_type', type=str, default="score",
                    help="reward type should in [score_empty_higher, score, higher, empty_higher]")
parser.add_argument('--visual', type=int, default=1,
                    help="visualize game board")
args = parser.parse_args()

game = environment.game_2048.Game2048(args)
# print(games.get_plat_state())
game.play_human()
