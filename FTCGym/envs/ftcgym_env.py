import gym
from gym import spaces
import numpy as np
import math
import sys
from six import StringIO
import matplotlib.pyplot as plt


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
        self.close_enough = 0.05
        self.distance = 0
        self.scatter_thickness = 4

        self.x = []
        self.y = []

        self.action_space = spaces.Box(low=self.lower_motor_bounds, high=self.upper_motor_bounds)

        self.lower_pos = np.array([0, 0])
        self.upper_pos = np.array([12, 12])

        self.observation_space = spaces.Box(low=self.lower_pos, high=self.upper_pos)
        self._randomize_starting()
        self.x = np.append(self.x, self.pos[0])
        self.y = np.append(self.y, self.pos[1])

    def step(self, action):

        assert self.action_space.contains(action)

        v = self._sum_velocity(action)
        reward = 0
        self.pos = np.round(np.add(self.pos, v), 2)
        ob = self.pos
        self.distance = np.round(math.sqrt(
                                 math.fabs((self.target[1]-self.pos[1])**2 -
                                           (self.target[0]-self.pos[0])**2)), 2)

        if self.distance > self.close_enough:
            reward = 1 / self.distance
        else:
            self.should_reset = True

        self.x = np.append(self.x, self.pos[0])
        self.y = np.append(self.y, self.pos[1])

        if self.pos[0] < 0 or self.pos[0] > 12 or self.pos[1] < 0 or self.pos[1] > 12:
            self.should_reset = True

        return ob, reward, self.should_reset, {str(self.distance)+" inches away."}

    def reset(self):

        self._randomize_starting()
        self.should_reset = False
        return self.pos

    def render(self, mode='human'):

        x_dots = np.delete(self.x, 0)
        y_dots = np.delete(self.y, 0)

        plt.plot(x_dots, y_dots)
        plt.plot(self.target[0], self.target[1], 'ro')
        plt.axis([0, 12, 0, 12])

        plt.xlabel('x (inch)')
        plt.ylabel('y (inch)')
        plt.title('Will it work?')
        plt.grid()

        plt.show()
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
