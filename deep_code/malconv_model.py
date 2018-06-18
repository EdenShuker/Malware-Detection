import torch
import torch.nn as nn


class MalConv(nn.Module):
    def __init__(self, labels, input_length=2000000, window_size=500):
        """
        :param labels: list of optional labels.
        :param input_length: length of input.
        :param window_size: size of window
        """
        super(MalConv, self).__init__()

        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(4, 128, window_size, stride=window_size, bias=True)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, len(labels))
        self.dropout = nn.Dropout(p=0.3)

        self.sigmoid = nn.Sigmoid()
        self.i2l = {i: l for i, l in enumerate(labels)}
        self.l2i = {l: i for i, l in self.i2l.iteritems()}

    def forward(self, x):
        """
        :param x: Variable input.
        :return: Variable output.
        """
        x = self.embed(x)
        # Channel first
        x = torch.transpose(x, -1, -2)

        cnn_value = self.conv_1(x.narrow(-2, 0, 4))
        gating_weight = self.sigmoid(self.conv_2(x.narrow(-2, 4, 4)))

        x = self.relu(cnn_value * gating_weight)
        x = self.pooling(x)
        x = self.dropout(x)

        x = x.view(-1, 128)
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)

        return x
