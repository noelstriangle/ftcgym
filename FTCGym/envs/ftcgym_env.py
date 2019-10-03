import gym
from gym import spaces
import numpy as np
import math
import sys
from six import StringIO


class FTCGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.pos = []
        self.target = []
        self.v = []
        self.scale = [0.02, 0.02]
        self.lower_motor_bounds = np.array([-1, -1, -1, -1])
        self.upper_motor_bounds = np.array([1, 1, 1, 1])
        self.should_reset = False
        self.close_enough = 100000
        self.distance = 0

        self.action_space = spaces.Box(low=self.lower_motor_bounds, high=self.upper_motor_bounds)

        self.lower_pos = np.array([0, 0])
        self.upper_pos = np.array([12, 12])

        self.observation_space = spaces.Box(low=self.lower_pos, high=self.upper_pos)
        self._randomize_starting()

    def step(self, action):

        assert self.action_space.contains(action)

        v = self._sum_velocity(action)
        self.pos = np.round(np.add(self.pos, v), 2)
        ob = self.pos
        self.distance = np.round(math.sqrt(
                                 math.fabs((self.target[1]-self.pos[1])**2 -
                                           (self.target[0]-self.pos[0])**2)), 2)
        reward = 1 / self.distance

        if reward > self.close_enough:
            self.should_reset = True

        return ob, reward, self.should_reset, {}

    def reset(self):

        self._randomize_starting()
        return self.pos

    def render(self, mode='human'):

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(str(self.pos))
        outfile.write("\n")
        outfile.write(str(self.distance))
        outfile.write("\n")
        return

    def _sum_velocity(self, action):
        fl = action[0]
        fr = action[1]
        bl = action[2]
        br = action[3]
        fl_vel = np.array([(fl * math.cos(math.pi / 4)), (fl * math.sin(math.pi / 4))])
        fr_vel = np.array([(fr * math.cos(math.pi * 3 / 4)), (fr * math.sin(math.pi * 3 / 4))])
        bl_vel = np.array([(bl * math.cos(math.pi * 3 / 4)), (bl * math.sin(math.pi * 3 / 4))])
        br_vel = np.array([(br * math.cos(math.pi / 4)), (br * math.sin(math.pi / 4))])
        f_vel = np.add(fl_vel, fr_vel)
        b_vel = np.add(bl_vel, br_vel)
        velocity = np.multiply(np.add(f_vel, b_vel), self.scale)
        return velocity

    def _randomize_starting(self):

        r = np.random.rand(2)
        self.pos = np.round(r*12, 2)
        r = np.random.rand(2)
        self.target = np.round(r*12, 2)
