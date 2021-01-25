from lgsvl_env.lgsvl_env import LgsvlEnv
import math, random
import numpy as np
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


def ddqn_learning(decay, gma, steps, train):
    env = LgsvlEnv()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = decay

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    model = nn.Sequential(
        nn.Linear(5, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    target_model = nn.Sequential(
        nn.Linear(5, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    optimizer = optim.Adam(model.parameters())

    replay_buffer = deque(maxlen=10000)

    def update_target(model, target_model):
        target_model.load_state_dict(model.state_dict())
    update_target(model, target_model)

    num_frames = steps
    batch_size = 4
    gamma = gma

    losses = []
    all_rewards = []
    episode_reward = 0
    a = []
    epoch = 0
    successes = 0

    if train:
        state = env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_by_frame(frame_idx)

            if random.random() > epsilon:
                st = torch.FloatTensor(state).unsqueeze(0)
                q_value = model(st)
                action = q_value.max(1)[1].data[0]
            else:
                action = random.randrange(2)

            next_state, reward, done, _ = env.step(int(action))

            st = np.expand_dims(state, 0)
            next_st = np.expand_dims(next_state, 0)
            replay_buffer.append((st, action, reward, next_st, done))

            episode_reward += reward

            d = (sqrt((env.x - 40) ** 2 + (env.z + 0) ** 2))
            print(epoch, frame_idx, state, int(action), reward, next_state, done, epsilon, d, successes)

            state = next_state

            if done:
                if episode_reward > 50:
                    successes += 1
                state = env.reset()
                all_rewards.append(episode_reward)
                a.append(frame_idx)
                episode_reward = 0
                epoch += 1


            if len(replay_buffer) > batch_size:
                st, action, rew, next_st, done = zip(*random.sample(replay_buffer, batch_size))
                st = np.concatenate(st)
                next_st = np.concatenate(next_st)

                st = torch.FloatTensor(np.float32(st))
                next_st = torch.FloatTensor(np.float32(next_st))
                action = torch.LongTensor(action)
                rew = torch.FloatTensor(rew)
                done = torch.FloatTensor(done)

                q_values = model(st)
                next_q_values = model(next_st)

                q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
                next_q_value = next_q_values.max(1)[0]
                expected_q_value = rew + gamma * next_q_value * (1 - done)

                loss = (q_value - expected_q_value.data).pow(2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.data)

        plt.plot(a, all_rewards)
        plt.savefig('reports/ddqn')

        return replay_buffer

