import torch
from torch.nn import Sequential, Conv2d, Linear, Embedding, LeakyReLU, MaxPool2d, functional


class Conv2(torch.nn.Module):
    def __init__(self, emb_type="emd") -> None:
        super().__init__()
        self.emb_type = emb_type
        assert emb_type in ["emd", "onehot"]

        if emb_type == "emd":
            self.embedding = Embedding(16, 64)
            self.conv1 = Conv2d(64, 128, (1, 2))
            self.conv2 = Conv2d(64, 128, (2, 1))
        elif emb_type == "onehot":
            self.conv1 = Conv2d(16, 128, (1, 2))
            self.conv2 = Conv2d(16, 128, (2, 1))

        self.conv3 = Conv2d(128, 128, (1, 2))
        self.conv4 = Conv2d(128, 128, (2, 1))
        self.conv5 = Conv2d(128, 128, (1, 2))
        self.conv6 = Conv2d(128, 128, (2, 1))
        self.activate_fn = LeakyReLU()
        self.fc = Sequential(
            Linear(7424, 256),
            LeakyReLU(),
            Linear(256, 4)
        )

    def forward(self, x):
        if self.emb_type == "emd":
            x = self.embedding(x)
        elif self.emb_type == "onehot":
            x = torch.nn.functional.one_hot(x, num_classes=16).float()
        x = x.permute(0, 3, 1, 2).contiguous()
        conv1 = self.activate_fn(self.conv1(x))
        conv2 = self.activate_fn(self.conv2(x))
        conv3 = self.activate_fn(self.conv3(conv1)).reshape(-1, 8 * 128)
        conv4 = self.activate_fn(self.conv4(conv1)).reshape(-1, 9 * 128)
        conv5 = self.activate_fn(self.conv5(conv2)).reshape(-1, 9 * 128)
        conv6 = self.activate_fn(self.conv6(conv2)).reshape(-1, 8 * 128)
        conv1_r = conv1.reshape(-1, 12 * 128)
        conv2_r = conv2.reshape(-1, 12 * 128)
        conv = torch.cat([conv1_r, conv2_r, conv3, conv4, conv5, conv6], dim=-1)
        output = self.fc(conv)
        return output


class CNNPool(torch.nn.Module):
    def __init__(self, emb_type="emd") -> None:
        super().__init__()
        self.emb_type = emb_type
        assert emb_type in ["emd", "onehot"]

        if emb_type == "emd":
            self.embedding = Embedding(16, 64)
            self.conv1 = Conv2d(64, 128, (2, 2), padding=1)
        elif emb_type == "onehot":
            self.conv1 = Conv2d(16, 128, (2, 2), padding=1)

        self.pool1 = MaxPool2d((2, 2), (1, 1))
        self.conv2 = Conv2d(128, 128, (2, 2), padding=1)
        self.pool2 = MaxPool2d((2, 2), (1, 1))
        self.conv3 = Conv2d(128, 128, (2, 2), padding=1)
        self.pool3 = MaxPool2d((2, 2), (1, 1))
        self.activate_fn = LeakyReLU()
        self.fc = Sequential(
            Linear(2048, 256),
            LeakyReLU(),
            Linear(256, 4)
        )

    def forward(self, x):
        if self.emb_type == "emd":
            x = self.embedding(x)
        elif self.emb_type == "onehot":
            x = torch.nn.functional.one_hot(x, num_classes=16).float()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.pool1(self.activate_fn(self.conv1(x)))
        x = self.pool2(self.activate_fn(self.conv2(x)))
        x = self.pool3(self.activate_fn(self.conv3(x)))
        x = torch.reshape(x, [-1, 2048])
        output = self.fc(x)
        return output


class CNN(torch.nn.Module):
    def __init__(self, emb_type="emd") -> None:
        super().__init__()
        self.emb_type = emb_type
        assert emb_type in ["emd", "onehot"]

        if emb_type == "emd":
            self.embedding = Embedding(16, 64)
            self.conv1 = Conv2d(64, 128, (3, 3), padding=1)
        elif emb_type == "onehot":
            self.conv1 = Conv2d(16, 128, (3, 3), padding=1)

        self.conv2 = Conv2d(128, 128, (3, 3), padding=1)
        self.conv3 = Conv2d(128, 128, (3, 3), padding=1)
        self.activate_fn = LeakyReLU()
        self.fc = Sequential(
            Linear(2048, 256),
            LeakyReLU(),
            Linear(256, 4)
        )

    def forward(self, x):
        if self.emb_type == "emd":
            x = self.embedding(x)
        elif self.emb_type == "onehot":
            x = torch.nn.functional.one_hot(x, num_classes=16).float()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.activate_fn(self.conv1(x))
        x = self.activate_fn(self.conv2(x))
        x = self.activate_fn(self.conv3(x))
        x = torch.reshape(x, [-1, 2048])
        output = self.fc(x)
        return output


class MultiHeadCNN(CNN):
    def __init__(self, emb_type="emd", n_heads=5):
        super(MultiHeadCNN, self).__init__(emb_type)
        self.n_heads = n_heads
        del self.fc
        self.fc = Sequential(
            Linear(2048, 256),
            LeakyReLU(),
            Linear(256, 4*n_heads)
        )

    def get_poisson_mask(self):
        """
        get mask
        :return: mask: [1, self.n_heads]
        """
        mask = torch.nn.Parameter(torch.poisson(torch.ones([1, self.n_heads])))
        return mask

    def forward(self, x):
        return super(MultiHeadCNN, self).forward(x)

    def get_value(self, states, head=None):
        head_values = self(states).reshape([-1, self.n_heads, 4])
        if head is not None:
            return head_values[:, head, :]
        else:
            return head_values

    def get_action(self, states):
        head_choices = self(states).reshape([-1, self.n_heads, 4]).argmax(dim=-1)  # [bsz, n_heads]
        target = torch.zeros(head_choices.shape[0], 4, dtype=head_choices.dtype, device=head_choices.device)
        values = torch.ones_like(head_choices)
        target.scatter_add_(1, head_choices, values)
        return torch.argmax(target, dim=-1)  # [bsz, n_heads]
