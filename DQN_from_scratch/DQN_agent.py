import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, fc1_dims=50, fc2_dims=50, path_dir='./'):
        super(QNetwork, self).__init__()

        self.checkpoint_file = os.path.join(path_dir, 'Q_Network_module')
        self.MLP = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        output = self.MLP(state)
        return output

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        T.load(self.state_dict)

class DQNMemory:
    def __init__(self, batch_size):
        self.states = []
        self.states_ = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batch(self):
        n_states = len(self.states)
        batch_start_indices = np.arange(0, n_states, self.batch_size)
        indices = np.arange(0, n_states)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start_indices]

        return np.array(self.states), np.array(self.states_), np.array(self.actions), \
               np.array(self.rewards), np.array(self.dones), batches

    def store_memory(self, state, state_, action, reward, done):
        self.states.append(state)
        self.states_.append(state_)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.states_ = []
        self.actions = []
        self.rewards = []
        self.dones = []


class Agent:
    def __init__(self, state_dims, action_dims, alpha, e_greedy, batch_size):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.e_greedy = e_greedy
        self.reply_buffer = DQNMemory(batch_size)
        self.Q_func = QNetwork(state_dims, action_dims, alpha)

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.Q_func.device)
        if np.random.rand() < self.e_greedy:
            action = np.random.choice(self.action_dims)
        else:
            action = int(T.argmax(self.Q_func(state)))
        return action

    def remember(self, state, state_, action, reward, done):
        self.reply_buffer.store_memory(state, state_, action, reward, done)

    def learn(self):
        pass

    def save(self):
        pass

    def load(self):
        pass