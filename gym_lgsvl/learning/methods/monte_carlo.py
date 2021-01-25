from lgsvl_env.lgsvl_env import LgsvlEnv
import pickle
import neptune
import matplotlib.pyplot as plt
import random


def monte_carlo(alpha, gma, epoch, train):
    epochs = []
    rewards = []
    if train:
        Q = []
        Q2 = Q
        env = LgsvlEnv()
        successes = 0
        for i in range(epoch):
            state = env.reset()
            R = []
            sr = len(state)
            epoch_reward = 0
            v = 0
            while True:
                flag = True
                for row in Q:
                    if state == row[:sr]:
                        current_step = [item for item in Q if item[:sr] == state][0]
                        if random.random()>0.1:
                            action = current_step [sr:].index(max(current_step[sr:]))
                        else:
                            action = random.randint(0, 1)
                        next_state, reward, done, info = env.step(action)
                        flag = False
                if flag == True:
                    row = state + [0, 0]
                    Q.append(row)
                    current_step = [item for item in Q if item[:sr] == state][0]
                    if random.random() > 0.1:
                        action = current_step[sr:].index(max(current_step[sr:]))
                    else:
                        action = random.randint(0, 1)
                    next_state, reward, done, info = env.step(action)

                f = []
                for j in range(len(current_step)):
                    if j == action + sr:
                        f.append(current_step[j] + reward)
                    else:
                        f.append(current_step[j])
                R.append(f)

                #current_step[action + sr] += reward
                state = next_state

                d = ('epoch: ' + str(i) + ' state:' + str(current_step[:sr]) + ' ' + str(current_step[sr:]) + ' action:' + str(
                    action) + ' reward:' + str(reward) + ' successes:' + str(successes))
                print(d)
                v += 1
                epoch_reward += reward

                if done:
                    if reward >= 199:
                        successes += 1
                    a = [0, 0]
                    for r in R:
                        for p in range(2):
                            a[p] += r[p + sr]
                    for d in range(2):
                        a[d] = (a[d] / v) * 0.01

                    for x in range(len(Q)):
                        for y in range(len(R)):
                            if Q[x][:sr] == R[y][:sr]:
                                for h in range(2):
                                    Q2[x][sr + h] += a[h]

                    Q = Q2


                    epochs.append(i)
                    rewards.append(epoch_reward)
                    neptune.log_metric('epochs_reward', epoch_reward)
                    with open('logs/log.pickle', 'wb') as f:
                        pickle.dump(d, f)
                    break

        plt.plot(epochs, rewards)
        plt.savefig('reports/mc')
        return Q

    else:
        with open('data/data_mc.pickle', 'rb') as f:
            Q = pickle.load(f)
        env = LgsvlEnv()
        for i in range(epoch):
            state = env.reset()
            sr = len(state)
            while True:
                current_step = [item for item in Q if item[:sr] == state][0]
                action = current_step[sr:].index(max(current_step[sr:]))
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    break
        return False