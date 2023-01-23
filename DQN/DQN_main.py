import gym
import numpy as np
import time
import torch
import moduls
from collections import deque
import random


class DQNAgent():

    def __init__(self, q_func, optimizer, n_acts, e_greed=0.1, gamma=0.9):
        self.q_func = q_func
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.e_greed = e_greed  # 用于设置智能体探索的概率
        self.n_acts = n_acts
        self.gamma = gamma

    def act(self, obs):
        if np.random.uniform(0, 1) < self.e_greed:  # 探索 exploration
            action = np.random.choice(self.n_acts)
        else:  # 利用 utilization
            action = self.predict(obs)

        return action

    def predict(self, obs):
        obs = torch.from_numpy(obs)
        obs = obs.to(torch.float32)
        Q_list = self.q_func(obs)
        # print(f'Q_list: {Q_list}')
        action = int(torch.argmax(Q_list).detach().numpy())
        # print(f'Predict: {action}')
        return action

    def learn(self, obs, action, reward, next_obs, done):
        """
        predict_Q: Q(obs, action)
        target_Q: reward + gamma * max(Q(next_obs))
        """
        obs, next_obs = torch.from_numpy(obs), torch.from_numpy(next_obs)
        obs, next_obs = obs.to(torch.float32), next_obs.to(torch.float32)
        # print(f'obs = {obs}')
        predict_Q = self.q_func(obs)[action]
        target_Q = reward + self.gamma * self.q_func(next_obs).max()*(1-float(done))
        # print(f'self.q_func(obs) = {self.q_func(obs)}')
        # print(f'self.q_func(next_obs) = {self.q_func(next_obs)}')
        # print(f'predict_Q = {predict_Q}; target_Q = {target_Q}')

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        # print(f'loss: {loss.item()}')
        loss.backward()
        self.optimizer.step()

    def learn_batch(self, batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)

        obs_batch = torch.FloatTensor(np.array(obs_batch))
        action_batch = torch.FloatTensor(np.array(action_batch)).to(torch.int64)
        reward_batch = torch.FloatTensor(np.array(reward_batch))
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        done_batch = torch.FloatTensor(np.array(done_batch))

        # predict Q
        predict_Q = self.q_func(obs_batch)
        action_onehot = torch.nn.functional.one_hot(action_batch, self.n_acts)
        predict_Q = (predict_Q * action_onehot).sum(1)

        # target Q
        next_obs_Qs = self.q_func(next_obs_batch)
        next_best_Q = next_obs_Qs.max(1)[0]
        target_Q = reward_batch + self.gamma * next_best_Q * (1 - done_batch)

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        # print(f'loss: {loss.item()}')
        loss.backward()
        self.optimizer.step()


class TrainManager():

    def __init__(self, env, episodes=1000, lr=0.001, gamma=0.9, e_greed=0.1, bufferCap=1000, start_bufferCap=200):
        self.env = env
        self.episodes = episodes
        obs_size = env.observation_space.shape[0]
        n_acts = env.action_space.n
        self.q_func = moduls.MPL(obs_size, n_acts)
        optimizer = torch.optim.AdamW(self.q_func.parameters(), lr=lr)
        self.agent = DQNAgent(
            q_func=self.q_func,
            optimizer=optimizer,
            n_acts=n_acts)
        self.replay_buffer = ExperienceReplay(bufferCap)
        self.start_bufferCap = start_bufferCap
        # self.model_save_file = f'D:/RL/saved_model/DQN_{time.strftime("%Y%m%d%H%M")}.pth'
        self.batch_size = 32

    def train(self):
        for episode in range(self.episodes):
            ep_reward = self.train_episode()
            print(f'Episode {episode}: ep_reward={ep_reward}')
            if episode % 100 == 99:
                ep_reward = self.test_episode()
                torch.save(self.q_func.state_dict(), f'D:/RL/saved_model/DQN_episode{episode}_reward{ep_reward}.pth')

    def train_episode(self):
        state = self.env.reset()
        total_reward = 0
        while True:
            action = self.agent.act(state)
            next_state, reward, done, _ = env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            if self.replay_buffer.__len__() > self.start_bufferCap:
                for i in range(5):
                    batch = self.replay_buffer.sample(self.batch_size)
                    self.agent.learn_batch(batch)
                    # state, action, reward, next_state, done

            state = next_state
            total_reward += reward

            if done: break

        return total_reward

    def test_episode(self):
        state = self.env.reset()
        total_reward = 0

        while True:
            action = self.agent.predict(state)
            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

            self.env.render()
            time.sleep(0.1)

            if done: break

        print(f'total reward = {total_reward}')
        return total_reward

    def evaluate_model(self, file_name):
        self.load_pretrain_model(file_name)
        self.test_episode()

    def load_pretrain_model(self, file_name):
        self.q_func.load_state_dict(torch.load(f'D:/RL/saved_model/{file_name}'))

class ExperienceReplay():

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        '''
        :param args: state, action, reward, next_state, done
        '''
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    tm = TrainManager(env=env)
    # tm.load_pretrain_model(file_name='DQN_202210081406.pth')
    tm.train()
    # tm.evaluate_model('DQN_episode199_reward392.0.pth')