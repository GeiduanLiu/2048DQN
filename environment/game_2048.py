import numpy as np
from .visualization import GameBoard


class Counter:
    def __init__(self):
        self.clear_score = 0

    def update(self, score):
        self.clear_score += score

    def get_score(self) -> int:
        return self.clear_score


class Game2048(object):
    HEIGHT = 4
    WIDTH = 4

    def __init__(self, args):
        self.actions = ['w', 's', 'a', 'd']  # Op: move UP, move DOWN, move LEFT, move RIGHT
        self.n_actions = len(self.actions)
        self.n_features = np.array([Game2048.HEIGHT, Game2048.WIDTH])
        self.board = np.zeros(shape=[Game2048.HEIGHT, Game2048.WIDTH], dtype=np.int32)
        self.reward_type = args.reward_type
        assert self.reward_type in ["score_empty_higher", "score", "higher", "empty_higher"]
        self.n_step = 0
        self.score = 0
        self.reset()

        try:
            self.use_visual_board = bool(args.visual)
        except AttributeError:
            self.use_visual_board = False
        if self.use_visual_board:
            self.visual_game_board = GameBoard(self)

    def reset(self):
        init_places = np.random.choice(a=np.array([5, 6, 9, 10], dtype=np.int32), size=2, replace=False)
        init_digital = np.random.choice(a=np.array([2, 4], dtype=np.int32), size=2, replace=True)
        self.board = np.zeros(shape=[Game2048.HEIGHT, Game2048.WIDTH])
        self.n_step = 0
        self.score = 0
        for i in range(2):
            self.board[init_places[i] // Game2048.HEIGHT][init_places[i] % Game2048.WIDTH] = init_digital[i]

    def step(self, op):
        if op in [0, 1, 2, 3]:
            op = self.actions[op]
        if op not in self.actions:
            return 'op error'
        op_num = np.where(np.array(self.actions) == op)[0][0]
        next_board, clear_score = self._move(self.board.copy(), op_num)
        done = False

        score_reward = np.log2(clear_score + 1) / 10.0
        empty_reward = np.sum(np.array(next_board) == 0) - np.sum(np.array(self.board) == 0)
        higher_reward = np.log2(np.max(next_board)) / 10.0 if np.max(next_board) != np.max(self.board) else 0

        if self.reward_type == "score_empty_higher":
            reward = score_reward + empty_reward + higher_reward
        elif self.reward_type == "score":
            reward = score_reward
        elif self.reward_type == "higher":
            reward = higher_reward
        elif self.reward_type == "empty_higher":
            reward = higher_reward + empty_reward
        else:
            raise RuntimeError("reward_type should in [score_empty_higher, score, higher, empty_higher]")
        self.score += clear_score
        if self._terminal(next_board):
            done = True

        if not done:
            #  Generate new position randomly
            if np.sum(next_board == 0) == 0:
                done = False
                return next_board, reward, done
            tmp = np.random.choice(a=len(np.where(next_board == 0)[0]), size=1)[0]
            xx = np.where(next_board == 0)[0][tmp]
            yy = np.where(next_board == 0)[1][tmp]
            new_score = 2 if np.random.uniform() < 0.9 else 4
            next_board[xx][yy] = new_score
            self.board = next_board.copy()
            self.n_step += 1
        return next_board, reward, done

    @staticmethod
    def has_score(board, action):
        tmp_board, _ = Game2048._move(board.copy(), action)
        return np.sum(np.abs(tmp_board - board))

    def _terminal(self, board):
        for i in range(self.n_actions):
            if self.has_score(board, i):
                return False
        return True

    @staticmethod
    def _line_squeeze(line, inv):
        # push box
        i, j = 3 if inv else 0, 3 if inv else 0
        d = -1 if inv else 1
        while (j >= 0) if inv else (j < 4):
            if line[j] != 0:
                t = line[j]
                line[j] = 0
                line[i] = t
                i += d
            j += d
        return line

    @staticmethod
    def _line_combine(line, inv, clear_score: Counter):
        # combine box with same value
        i = 3 if inv else 0
        d = -1 if inv else 1
        while (i >= 1) if inv else (i < 3):
            if line[i] != 0 and line[i] == line[i + d]:
                line[i] += line[i + d]
                clear_score.update(line[i])
                line[i + d] = 0
                i += d
            i += d
        return line

    @staticmethod
    def _move(board, op_num):
        clear_score = Counter()
        inv = (op_num % 2 == 1)  # inverse direction

        def f(line, _clear_score):
            return Game2048._line_squeeze(Game2048._line_combine(Game2048._line_squeeze(line, inv), inv, _clear_score), inv)

        for i in range(4):
            if op_num < 2:
                board[:, i] = f(board[:, i], clear_score)
            else:
                board[i, :] = f(board[i, :], clear_score)
        return board, clear_score.get_score()

    def get_plat_state(self):
        return self.board.reshape([-1, ])

    def get_state(self):
        return self.board.copy()

    def get_score(self):
        return self.score

    def show_game(self):
        print(self.board)

    def play_human(self):
        if self.use_visual_board:
            self.visual_game_board.start()
        else:
            while True:
                self.show_game()
                digit = input()
                print('your op:', digit)
                if digit == 'q':
                    break
                elif digit == 'p':
                    self.reset()
                elif digit in self.actions:
                    self.step(digit)
                    print('score:', self.get_score())
                else:
                    print('invalid input')
