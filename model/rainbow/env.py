# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

from environment.game_2048 import Game2048


class Args():
    pass


class Env():
    def __init__(self, args):
        self.device = args.device

        game_args = Args()
        game_args.visual = 0
        game_args.reward_type = "score"     # todo
        self.game = Game2048(game_args)

        self.actions = dict((i, e) for i, e in enumerate(self.game.actions))
        self.training = True  # Consistent with model training mode  # todo: useless in game2048?

    def _get_state(self):
        state = self.game.get_state()
        state = np.floor(np.log2(state + 1)).astype(np.int8)
        return torch.tensor(state, dtype=torch.uint8, device=self.device)

    def reset(self):
        # Reset internals
        self.game.reset()
        # Process and return "initial" state
        observation = self._get_state()
        return torch.stack([observation], 0)

    def step(self, action):
        _, reward, done = self.game.step(self.actions.get(action))
        observation = self._get_state()
        # Return state, reward, done
        return torch.stack([observation], 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        print(self.game.get_state())

    def close(self):
        pass
