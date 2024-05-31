import torch
import torch.nn as nn
import numpy as np


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, filename):
        super().__init__()
        self.input_dim = input_dim
        channels = input_dim[0]

        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        conv_output_size = self._conv_output_dim()
        lin1_output_size = 512

        self.l2 = nn.Sequential(
            nn.Linear(conv_output_size, lin1_output_size),
            nn.ReLU(),
            nn.Linear(lin1_output_size, output_dim),
        )

        self.filename = filename

    # Calculates output dimension of conv layers
    def _conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], -1)
        actions = self.l2(x)

        return actions

    def save_model(self):
        torch.save(self.state_dict(), "./models/" + self.filename + ".pth")

    def load_model(self):
        self.load_state_dict(torch.load("./models/" + self.filename + ".pth"))
