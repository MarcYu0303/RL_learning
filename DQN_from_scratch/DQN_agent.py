import torch as T
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, fc1_dims=50,
                 fc2_dims=50, path_dir='./'):
        super(QNetwork, self).__init__()

        self.checkpoint_file = os.path.join(path_dir, 'Q_Network_module.ckpt')
        self.MLP = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
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
        self.load_state_dict(T.load(self.checkpoint_file))


class DQNMemory:
    def __init__(self, batch_size, max_length = 1000):
        self.states = deque([], maxlen=max_length)
        self.states_ = deque([], maxlen=max_length)
        self.actions = deque([], maxlen=max_length)
        self.rewards = deque([], maxlen=max_length)
        self.dones = deque([], maxlen=max_length)

        self.batch_size = batch_size
        self.max_length = max_length

    def generate_batch(self):
        n_states = len(self.states)
        indices = np.arange(0, n_states)
        batch = np.random.sample(indices, self.batch_size)

        return np.array(self.states), np.array(self.states_), np.array(self.actions), \
               np.array(self.rewards), np.array(self.dones), batch

    def store_memory(self, state, state_, action, reward, done):
        self.states.append(state)
        self.states_.append(state_)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = deque([], maxlen=self.max_length)
        self.states_ = deque([], maxlen=self.max_length)
        self.actions = deque([], maxlen=self.max_length)
        self.rewards = deque([], maxlen=self.max_length)
        self.dones = deque([], maxlen=self.max_length)

class Agent:
    def __init__(self, state_dims, action_dims, alpha, e_greedy,
                 batch_size, n_epochs, gamma):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.e_greedy = e_greedy
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.reply_buffer = DQNMemory(batch_size)
        self.Q_func = QNetwork(state_dims, action_dims, alpha)
        self.loss = nn.MSELoss()
        self.device = self.Q_func.device

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
        for _ in range(self.n_epochs):
            states_arr, states_arr_, actions_arr, rewards_arr, \
            dones_arr, batch = self.reply_buffer.generate_batch()

            states, states_ = T.tensor(states_arr, dtype=T.float).to(self.device), \
            T.tensor(states_arr_, dtype=T.float).to(self.device)
            rewards = T.tensor(rewards_arr, dtype=T.float).to(self.device)

            q = self.Q_func(states[batch])
            q_ = self.Q_func(states_[batch])
            TD_target = rewards[batch] + self.gamma * q_
            Loss = self.loss(q, TD_target).to(self.device)
            self.Q_func.optimizer.zero_grad()
            Loss.backward()
            self.Q_func.optimizer.step()

        self.reply_buffer.clear_memory()

    def save(self):
        self.Q_func.save_checkpoint()

    def load(self):
        self.Q_func.load_checkpoint()
