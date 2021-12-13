import numpy as np
import torch
from torch.nn.modules import MSELoss
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm

from model.DQNPlayer import DQNPlayer
from model.Layers import MultiHeadCNN
from environment.game_2048 import Game2048


class BootstrapDQN(DQNPlayer):
    def __init__(self,
                 n_actions,
                 n_features,
                 n_heads,
                 model_dir,
                 args,
                 use_cuda=True,
                 ):
        super(BootstrapDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_heads = n_heads
        self.args = args
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.lr_decay_steps = args.lr_decay_steps
        self.log_steps = args.log_steps
        self.gamma = args.gamma
        self.memory_size = args.memory_size
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.use_cuda = use_cuda
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.episode = 0
        self.modeldir = model_dir
        self.model_type = args.model_type
        self.embedding_type = args.embedding_type
        self._build_net_cnn()
        # 存储空间下标：0-31:S, 32-63:S_, 64:action, 65:reward
        self.memory = np.zeros((self.memory_size, self.n_features[0] * self.n_features[1] * 2 + 2))
        self.cur_head = self._choose_head()

    def _choose_head(self):
        return np.random.randint(0, self.n_heads)

    def update_episode(self, episode, *args, **kwargs):
        if episode != self.episode:
            self.cur_head = self._choose_head()
            self.episode = episode

    @staticmethod
    def preprocess_state(state):  # Preprocess
        return torch.from_numpy(np.floor(np.log2(state + 1)).astype("int64")).to(torch.long)

    @staticmethod
    def _remove_impossible(action_values, board, bsz_idx, act_idx):
        if not Game2048.has_score(board, act_idx):
            if len(action_values.shape) == 3:
                action_values[bsz_idx, :, act_idx] = torch.min(action_values[bsz_idx, :, :]) - 10
            else:
                action_values[bsz_idx, act_idx] = torch.min(action_values[bsz_idx, :]) - 10

    def choose_action(self, state, *args, det=True, **kwargs):
        """
        Now it can handle a batch of states

        :param state: np.array(2-D) or a list of states [np.array(2-D)]
        :param det:
        :return:
        """
        if type(state) is list:
            tstate = np.asarray(state)
        else:
            tstate = state[np.newaxis, :, :]
        tensor_tstate = self.preprocess_state(tstate).to('cuda' if self.use_cuda else 'cpu')
        if det and np.random.uniform() < self.epsilon:
            action_value = np.asarray([[np.random.random() if Game2048.has_score(tstate[j, :, :], i) else -1
                                        for i in range(4)] for j in range(tstate.shape[0])])
            action_index = np.argmax(action_value, axis=-1)
        else:
            if det:
                head = self.cur_head
            else:
                head = None
            with torch.no_grad():
                action_value = self.q_eval_model.get_value(tensor_tstate, head)
            action_value = action_value.detach().cpu()  # [bsz, n_heads, 4] or [bsz, 4]
            [[self._remove_impossible(action_value, tstate[j, :, :], j, i) for i in range(4)]
             for j in range(tstate.shape[0])]
            if len(action_value.shape) == 2:
                action_index = torch.argmax(action_value, dim=-1).numpy()
            else:
                head_choices = torch.argmax(action_value, -1)
                target = torch.zeros(head_choices.shape[0], 4, dtype=head_choices.dtype, device=head_choices.device)
                values = torch.ones_like(head_choices)
                target.scatter_add_(1, head_choices, values)
                action_index = torch.argmax(target, dim=-1).numpy()  # [bsz]

        action_index = [np.squeeze(e, 0) for e in np.split(action_index, action_index.shape[0], axis=0)]
        if type(state) is list:
            return action_index
        else:
            return action_index[0]

    def _build_net_cnn(self):

        self.q_eval_model = MultiHeadCNN(self.embedding_type, self.n_heads)\
            .to('cuda' if self.use_cuda else 'cpu')
        self.q_target_model = MultiHeadCNN(self.embedding_type, self.n_heads)\
            .to('cuda' if self.use_cuda else 'cpu')

        self.q_target_model.eval()
        self.q_eval_model.eval()
        self.opt = Adam(self.q_eval_model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.opt, step_size=self.lr_decay_steps, gamma=self.lr_decay)
        self.loss = MSELoss()

    def _fit(self, input_states, update_actions, target_q):
        """
        training body

        :param input_states:
        :param update_actions: [bsz]
        :param target_q: [bsz, n_heads]
        :return:
        """
        output = target_q.detach()
        update_actions = torch.tile(update_actions.detach().unsqueeze(-1), (1, self.q_eval_model.n_heads)) \
            .reshape([-1])
        action_values = self.q_eval_model(input_states).reshape([-1, 4])
        pred = action_values[np.arange(action_values.shape[0], dtype=np.int32), update_actions]
        pred = torch.reshape(pred, [self.batch_size, self.q_eval_model.n_heads])
        mask = self.q_eval_model.get_poisson_mask().to(pred.device)  # [1, n_heads]
        pred = torch.mul(pred, mask)  # [bsz, n_heads]
        # pred = self.q_eval_model(input_states)[np.arange(self.batch_size, dtype=np.int32), update_actions]
        ploss = self.loss(pred, output) / self.q_eval_model.n_heads
        self.opt.zero_grad()
        ploss.backward()
        self.opt.step()
        self.scheduler.step()

        if self.learn_step_counter % self.log_steps == 0:
            tqdm.write(f'Episode: {self.episode}; learn_step: {self.learn_step_counter}; loss: {ploss.item()}; '
                       f'lr: {self.opt.param_groups[0]["lr"]}, "epsilon": {self.epsilon}')

    def target_replace_op(self):
        p1 = self.q_eval_model.state_dict()
        self.q_target_model.load_state_dict(p1)
        self.q_target_model.eval()

    def store_memory(self, s, s_, a, r):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        s = self.preprocess_state(s)
        s_ = self.preprocess_state(s_)
        memory = np.hstack((s, s_, [a, r]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self, n_epoch):
        self.q_eval_model.train()
        for _ in range(n_epoch):
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            n_features = self.n_features[0] * self.n_features[1]

            s = torch.LongTensor(
                batch_memory[:, 0:n_features].reshape([-1, self.n_features[0], self.n_features[1]])).to(
                'cuda' if self.use_cuda else 'cpu')
            s_ = torch.LongTensor(
                batch_memory[:, n_features:n_features * 2].reshape([-1, self.n_features[0], self.n_features[1]])).to(
                'cuda' if self.use_cuda else 'cpu')
            a = torch.Tensor(batch_memory[:, n_features * 2]).long().to('cuda' if self.use_cuda else 'cpu')
            r = torch.Tensor(batch_memory[:, n_features * 2 + 1]).to('cuda' if self.use_cuda else 'cpu')

            with torch.no_grad():
                q_next = self.q_target_model.get_value(s_)  # [bsz, n_heads, 4]

            r = torch.unsqueeze(r, -1)
            q_target = r + self.gamma * torch.max(q_next, dim=-1)[0]  # [bsz, n_heads]

            self._fit(s, a, q_target)

            self.learn_step_counter += 1
        self.q_eval_model.eval()

    def save_model(self, episode):
        torch.save(self.q_eval_model.state_dict(), '{}/2048-{}.h5'.format(self.modeldir, episode))

    def load_model(self, model_path):
        self.q_eval_model.load_state_dict(
            torch.load(model_path, map_location=lambda a, b: a if self.use_cuda is False else None))
