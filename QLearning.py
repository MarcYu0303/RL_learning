import gym
import numpy as np
import gridworld
import time

class QlearningAgent():

    def __init__(self, n_states, n_acts, e_greed=0.1, lr=0.1, gamma=0.9):
        self.Q = np.zeros((n_states, n_acts))
        self.e_greed = e_greed  # 用于设置智能体探索的概率
        self.n_acts = n_acts
        self.lr = lr
        self.gamma = gamma

    def act(self, state):
        if np.random.uniform(0, 1) < self.e_greed:  # 探索 exploration
            action = np.random.choice(self.n_acts)
        else:  # 利用 utilization
            action = self.predict(state)

        return action

    def predict(self, state):
        Q_list = self.Q[state, :]
        # np.argmax(Q_list)  # 返回最大值索引 但初始状态[0, 0, 0, 0] return: 0
        action = np.random.choice(np.flatnonzero(Q_list == Q_list.max()))  # better choice
        return action

    def learn(self, state, action, reward, next_state, done):
        """
        Q(St, At) <- Q(St, At) + lr * (TD_target - Q(St, At)))
        TD_target = Rt + gamma * max(Q(S(t+1))
        St: current state
        At: current action
        Rt+1: next state reward (get from interaction between At and env)
        St+1: next state (get from interaction between At and env)
        max(Q(S(t+1)): maximum Q value of S(t+1)
        """
        cur_Q = self.Q[state, action]

        if done:
            target_Q = reward
        else:
            Q_list = self.Q[next_state, :]
            maxQ_action = np.random.choice(np.flatnonzero(Q_list == Q_list.max()))
            target_Q = reward + self.gamma * self.Q[next_state, maxQ_action]

        self.Q[state, action] += self.lr * (target_Q - cur_Q)

    def get_Qtable(self):
        return self.Q


def train(env, episodes=500, lr=0.1, gamma=0.9, e_greed=0.1):
    agent = QlearningAgent(
        n_states=env.observation_space.n,
        n_acts=env.action_space.n,
        lr=lr,
        gamma=gamma,
        e_greed=e_greed)

    show_render = False
    for episode in range(episodes):
        ep_reward = train_episode(env, agent, show_render)
        print(f'Episode {episode}: total reward = {ep_reward}')

        # if episode%100 == 0:
        #     show_render = True
        # else:
        #     show_render = False

    test_episode(env, agent, show_render=True)
    # print(agent.get_Qtable())

def train_episode(env, agent, show_render):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        agent.learn(state, action, reward, next_state, done)

        state = next_state

        total_reward += reward

        if show_render:
            print(f'reward = {reward}')
            env.render()
            time.sleep(0.5)

        if done: break
    return total_reward


def test_episode(env, agent, show_render):
    total_reward = 0
    state = env.reset()

    while True:
        action = agent.predict(state)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward

        if show_render:
            print(f'action = {action}; reward = {reward}')
            env.render()
            time.sleep(0.5)

        if done: break
    print(f'total reward = {total_reward}')
    return reward

if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = gridworld.CliffWalkingWapper(env)
    train(env)
