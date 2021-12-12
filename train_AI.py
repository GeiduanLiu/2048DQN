import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from environment.game_2048 import Game2048
from model import DeepQNetwork


def evaluate(train_episode, times):
    print("evaluate")
    score_list = []
    max_tile_list = []
    for episode in tqdm(range(times)):
        game.reset()
        s = game.get_state()
        while True:
            action_index = RL.choose_action(s, det=False)
            s_, r, done = game.step(action_index)
            s = s_
            if done:
                score_list.append(int(game.get_score()))
                max_tile_list.append(int(np.max(s_)))
                break

    output_str = "episode: %s, avg_score: %.2f, " % (str(train_episode), sum(score_list) / times)
    for max_tile in [4096, 2048, 1024, 512, 256]:
        output_str += "max_tile %d rate: %.2f%%, " % (max_tile, max_tile_list.count(max_tile) * 100 / times)
    print(output_str)


def test_2048(args):
    RL.load_model(args.test_model)
    evaluate("test", args.test_times)


def train_2048(args):
    step = 0
    for episode in range(args.episode):
        RL.episode = episode
        if (episode + 1) % args.target_replace_episode == 0:
            RL.target_replace_op()
        game.reset()
        s = game.get_state()
        game_step = 0
        while True:
            action_index = RL.choose_action(s)
            s_, r, done = game.step(action_index)
            # print('action:', game.actions[action_index])
            # print('game:\n', s_, '\n')
            if episode > 10000 or (RL.epsilon > 0.1 and step % 2500 == 0):
                RL.epsilon = RL.epsilon / 1.005
            RL.store_memory(s.reshape([-1, ]), s_.reshape([-1, ]), action_index, r)

            s = s_
            if done:
                break
            step += 1
            game_step += 1

        if episode > args.start_train_episode:
            for i in range(args.train_epoch):
                RL.learn()

        if episode > args.start_evaluate_episode and (episode + 1) % args.evaluate_episode == 0:
            RL.save_model(episode + 1)
            evaluate(episode, args.evaluate_times)

    print('game over')
    RL.save_model(episode + 1)
    print('model saved!')


if __name__ == '__main__':
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

    game = Game2048(args)

    RL = DeepQNetwork(n_actions=game.n_actions,
                      n_features=game.n_features,
                      gameref=game,
                      use_cuda=use_cuda,
                      model_dir=model_dir,
                      args=args)

    if args.mode == "train":
        train_2048(args)
    elif args.mode == "test":
        test_2048(args)
