import gym
import numpy as np

env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
lr = .85
y = .99
num_episodes = 10000
# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1

        # epsilon greedy action decision
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        # env.render()

        # Update Q-Table with new exp
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if d == True:
            break

    rList.append(rAll)

print('Total reward: ', sum(rList))
# print('Q: ', Q)
