import gym
from DQN_agent import Agent
import numpy as np
from utils import plot_running_average_curve, plot_learning_curve


def main() -> None:
    env = gym.make('CartPole-v1')

    n_games = 1000
    n_steps_to_learn = 10
    batch_size = 5
    alpha = 0.001
    n_epochs = 3
    gamma = 0.9
    e_greedy = 0.1
    memory_start_size = 100

    agent = Agent(action_dims=env.action_space.n,
                  state_dims=env.observation_space.shape[0],
                  batch_size=batch_size,
                  alpha=alpha,
                  gamma=gamma,
                  e_greedy=e_greedy,
                  n_epochs=n_epochs)

    learn_iter = 0
    best_score = env.reward_range[0]
    score_history = []
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.reply_buffer.store_memory(observation, observation_,
                                            action, reward, done)
            score += reward
            n_steps += 1
            if n_steps % n_steps_to_learn == 0 and memory_start_size < agent.reply_buffer.__len__():
                agent.learn()
                learn_iter += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print(f'episode {i}, score {score}, avg_score {round(avg_score, 2)}, '
              f'time steps {n_steps}, learning iterations {learn_iter}')
    x = np.arange(1, len(score_history) + 1)
    plot_running_average_curve(x, score_history, './plots/running_average.png')




if __name__ == '__main__':
    main()