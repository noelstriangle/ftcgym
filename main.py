import gym
import FTCGym
import numpy as np

env = gym.make("ftcgym-v0")

env.reset()

for _ in range(1, 10):
    env.step(env.action_space.sample())
    env.render()

env.close()
