import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('FrozenLake-v0')

inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
updateModel = trainer.minimize(loss)

y = .99
e = 0.5
num_episodes = 5000

totalRewardsList = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        totalReward = 0
        done = False
        game_n = 0
        # The Q-Network
        while game_n < 99:
            game_n += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            action, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[state:state + 1]})
            if np.random.rand(1) < e:
                action[0] = env.action_space.sample()

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action[0])

            # Obtain the Q' values by feeding the new state through our network
            Qpredicted = sess.run(Qout, feed_dict={inputs1: np.identity(16)[new_state:new_state + 1]})

            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Qpredicted)

            targetQ = allQ
            targetQ[0, action[0]] = reward + y * maxQ1

            # Train our network using target and predicted Q values
            _, _ = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[state:state + 1], nextQ: targetQ})
            totalReward += reward
            state = new_state
            if done:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                break
        totalRewardsList.append(totalReward)
        print('Avg. reward: ', sum(totalRewardsList)/(1+i))