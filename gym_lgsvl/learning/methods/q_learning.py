from lgsvl_env.lgsvl_env import LgsvlEnv
import pickle
import neptune
import matplotlib.pyplot as plt

def q_learning(alpha, gma, epoch, train):
    x = []
    y = []
    if train:
        successes = 0
        Q = []
        env = LgsvlEnv()
        for i in range(epoch):
            state = env.reset()
            sr = len(state)
            epoch_reward = 0
            while True:
                flag = True
                for row in Q:
                    if state == row[:sr]:
                        current_step = [item for item in Q if item[:sr] == state][0]
                        action = current_step[sr:].index(max(current_step[sr:]))
                        next_state, reward, done, info = env.step(action)
                        flag = False
                if flag == True:
                    row = state + [0, 0]
                    Q.append(row)
                    current_step = [item for item in Q if item[:sr] == state][0]
                    action = current_step[sr:].index(max(current_step[sr:]))
                    next_state, reward, done, info = env.step(action)

                flag = True
                for row in Q:
                    if next_state == row[:sr]:
                        next_step = [item for item in Q if item[:sr] == next_state][0]
                        best_next_action = next_step[sr:].index(max(next_step[sr:]))
                        flag = False
                if flag == True:
                    row = next_state + [0, 0]
                    Q.append(row)
                    next_step = [item for item in Q if item[:sr] == next_state][0]
                    best_next_action = next_step[sr:].index(max(next_step[sr:]))

                current_step[action + sr] = current_step[action + sr] + alpha * (
                            reward + gma * (next_step[best_next_action + sr]) - current_step[action + sr])

                state = next_state
                epoch_reward += reward
                d = ('epoch: ' + str(i) + ' state:' + str(current_step[:sr]) + ' ' + str(
                    current_step[sr:]) + ' action:' + str(
                    action) + ' reward:' + str(reward) + ' successes:' + str(successes))
                print(d)
                if done:
                    if reward >= 199:
                        successes += 1
                    x.append(i)
                    y.append(epoch_reward)
                    neptune.log_metric('epochs_reward', epoch_reward)
                    with open('logs/log.pickle', 'wb') as f:
                        pickle.dump(d, f)
                    break

        plt.plot(x, y)
        plt.savefig('reports/ql')
        return Q
    else:
        with open('data/data.pickle', 'rb') as f:
            Q = pickle.load(f)
        env = LgsvlEnv()
        for i in range(epoch):
            state = env.reset()
            sr =len(state)
            while True:
                current_step = [item for item in Q if item[:sr] == state][0]
                action = current_step[sr:].index(max(current_step[sr:]))
                next_state, reward, done, info = env.step(action)
                state = next_state
                if done:
                    break
        return False