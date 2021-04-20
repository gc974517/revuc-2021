import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQLearner(nn.Module):
    def __init__(self, board_size, hidden_size, state_size, actions, layers):
        super(DQLearner, self).__init__()
        self.encoder = Encoder(board_size, hidden_size, state_size)
        self.Q = DQN(state_size, actions, hidden_size, layers)

    def forward(self, x):
        return self.Q(self.encoder(x))

class DQN(nn.Module):
    def __init__(self, state, controls, hidden_size, layers):
        super(DQN, self).__init__()
        self.input = nn.Linear(state, hidden_size)
        self.hidden = [nn.Linear(hidden_size, hidden_size) for i in range(layers)]
        self.output = nn.Linear(hidden_size, controls)

    def forward(self, x):
        x = F.relu(self.input(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output(x)

class Encoder(nn.Module):
    def __init__(self, board_size, hidden_size, state_size):
        super(Encoder, self).__init__()
        self.w_1 = nn.Linear(board_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, state_size)

    def forward(self, x):
        x = torch.Tensor(x).flatten().unsqueeze(0)

        x = self.w_1(x)
        x, _ = attention(x, x, x)
        return self.w_2(x)

def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / np.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

def make_model(board_size, hidden_size, state_size, actions, layers):
    model = DQLearner(board_size, hidden_size, state_size, actions, layers)
    return model