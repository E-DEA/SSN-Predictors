#!/usr/bin/env python3

import torch.nn as nn

criterion = nn.MSELoss()

class FFNN(nn.Module):
    def __init__(self, inp_dim=6):
        super(FFNN, self).__init__()

        layers = []
        layers += [ nn.Linear(inp_dim, 6),
                    nn.LeakyReLU(inplace=True)]
        layers += [ nn.Linear(6, 3),
                    nn.LeakyReLU(inplace=True) ]
        layers += [ nn.Linear(3, 1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x

class RNN(nn.Module):
    def __init__(self, inp_dim=6):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(inp_dim, 8, 2, batch_first=True)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.out(out)

        return out
