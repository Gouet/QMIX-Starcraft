import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim=64, n_actions=1):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        print('input_shape: ', input_shape)

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    def update(self, agent):
        for param, target_param in zip(agent.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
