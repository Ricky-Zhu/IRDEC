from gym import Wrapper
import numpy as np


class StepWrapperAntMaze(Wrapper):  # for wrapper the roboverse env to return a vector state
    def __init__(self, env):
        super(StepWrapperAntMaze, self).__init__(env)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        s = obs['observation']
        r = 0.

        if info['is_success']:
            r = 1.
        done = False

        return s, r, done, info

    def reset(self, **kwargs):
        s = self.env.reset()['observation']
        return s
