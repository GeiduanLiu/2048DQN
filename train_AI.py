import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from environment.game_2048 import Game2048
from model.DeepQNetwork import DeepQNetwork


def evaluate(player, game, train_episode, times):
    tqdm.write("evaluate")
    score_list = []
    max_tile_list = []
    for _ in range(times):
        game.reset()
        s = game.get_state()
        while True:
            action_index = player.choose_action(s, det=False)
            s_, r, done = game.step(action_index)
            s = s_
            if done:
                score_list.append(int(game.get_score()))
                max_tile_list.append(int(np.max(s_)))
                break

    output_str = "episode: %s, avg_score: %.2f, " % (str(train_episode), sum(score_list) / times)
    for max_tile in [4096, 2048, 1024, 512, 256]:
        output_str += "max_tile %d rate: %.2f%%, " % (max_tile, max_tile_list.count(max_tile) * 100 / times)
    tqdm.write(output_str)


def test_2048(player, game, args):
    player.load_model(args.test_model)
    evaluate(player, game, "test", args.test_times)


def train_2048(player, games, eval_game, args):
    """
    Training process

    :param player: the DQN network
    :param games:  A list of games generator for playing in parallel
    :param eval_game
    :param args:
    :return:
    """
    n_games = len(games)
    print(f"{n_games} game running in parallel")
    step = 0
    epsilon_update_phase = 0
    weight_update_phase = 0
    evaluate_update_phase = 0
    episode = 0
    player.episode = 0
    [game.reset() for game in games]
    states = [game.get_state() for game in games]

    with tqdm(total=args.episode, desc="episode") as pbar:
        while episode < args.episode:
            # play a step across the games
            action_index = player.choose_action(states)
            updates = [game.step(index) for game, index in zip(games, action_index)]  # (s_, r, done)
            [player.store_memory(s.reshape([-1, ]), s_.reshape([-1, ]), action, r)
             for s, action, (s_, r, done) in zip(states, action_index, updates)]
            step += n_games

            # check if there's any games finished
            finished = [update[-1] for update in updates]
            n_finised = sum(finished)

            # update the episode counts
            episode += n_finised
            pbar.update(n_finised)
            player.episode = episode

            # update the epsilon
            if episode > 10000 or (player.epsilon > 0.1 and step // 2500 > epsilon_update_phase):
                player.epsilon = player.epsilon / 1.005
                epsilon_update_phase = step // 2500

            # Update the network if ok
            if episode > args.start_train_episode and n_finised > 0:
                for i in range(args.train_epoch):
                    player.learn()

            # Update the games states
            [game.reset() if finish else None for game, finish in zip(games, finished)]
            states = [s_ if done else game.get_state() for game, (s_, r, done) in zip(games, updates)]

            # update the memory
            if episode // args.target_replace_episode > weight_update_phase:
                player.target_replace_op()
                weight_update_phase = episode // args.target_replace_episode

            # evaluate
            if episode > args.start_evaluate_episode and episode // args.evaluate_episode > evaluate_update_phase:
                player.save_model(episode + 1)
                evaluate(player, eval_game, episode, args.evaluate_times)
                evaluate_update_phase = episode // args.evaluate_episode

    print('games over')
    player.save_model(episode)
    print('model saved!')


def main():
    parser = argparse.ArgumentParser(description='2048 DQN model')
    parser.add_argument('--mode', type=str, default="train", help="train or test")
    parser.add_argument('--test_model', type=str, default=None, help="test model")
    parser.add_argument('--test_times', type=int, default=10000, help='test times')
    parser.add_argument('--episode', type=int, default=200000, help="episode")
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.99, help='learning rate decay rate')
    parser.add_argument('--lr_decay_steps', type=int, default=10000, help='learning rate decay steps')
    parser.add_argument('--gamma', type=float, default=0.95, help='reward decay')
    parser.add_argument('--epsilon', type=float, default=0.9, help='epsilon greedy')
    parser.add_argument('--memory_size', type=int, default=6000, help='data memory size')
    parser.add_argument('--batch_size', type=int, default=512, help='train batch size')
    parser.add_argument('--target_replace_episode', type=int, default=100, help='target model update frequence')
    parser.add_argument('--evaluate_episode', type=int, default=500, help='evaluate frequence')
    parser.add_argument('--evaluate_times', type=int, default=10, help="evaluate times")
    parser.add_argument('--log_steps', type=int, default=1000, help='log frequence')
    parser.add_argument('--reward_type', type=str, default="score",
                        help="reward type should in [score_empty_higher, score, higher, empty_higher]")
    parser.add_argument('--model_type', type=str, default="Conv2", help="model type should in [Conv2, CNNPool, CNN]")
    parser.add_argument('--embedding_type', type=str, default="emd", help="embedding type should in [emd, onehot]")
    parser.add_argument('--train_epoch', type=int, default=10, help='training epochs per episode')
    parser.add_argument('--start_train_episode', type=int, default=300, help="when to start train")
    parser.add_argument('--start_evaluate_episode', type=int, default=1000, help="when to start evaluate")

    args = parser.parse_args()
    print(args)

    model_dir = "model/%s_%s" % (args.model_type, args.embedding_type)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    use_cuda = torch.cuda.is_available()

    games = [Game2048(args) for _ in range(32)]
    eval_game = Game2048(args)

    player = DeepQNetwork(n_actions=games[0].n_actions,
                          n_features=games[0].n_features,
                          use_cuda=use_cuda,
                          model_dir=model_dir,
                          args=args)

    if args.mode == "train":
        train_2048(player, games, eval_game, args)
    elif args.mode == "test":
        test_2048(player, eval_game, args)


if __name__ == '__main__':
    main()
