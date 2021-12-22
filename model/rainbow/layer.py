import torch
from torch.nn import Sequential, Conv2d, Linear, Embedding, LeakyReLU, MaxPool2d, functional


class Conv2Layer(torch.nn.Module):
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
        return conv
