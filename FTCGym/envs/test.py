from gym import spaces
import numpy as np

f = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))

action = f.sample()

print(action[0])
