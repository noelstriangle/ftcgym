import gym
import FTCGym
import numpy as np

env = gym.make("ftcgym-v0")
reward = 0
info = ''

for i_episode in range(5):
    observation = env.reset()
    for t in range(500):
        print(observation)
        print(reward)
        print(info)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} time-steps".format(t+1))
            break
    env.render()

env.close()
