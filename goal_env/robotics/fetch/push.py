from gym import utils
from goal_env.robotics import fetch_env
import numpy as np


class FetchPushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', no_fence=True):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'table0:slide0': 1.05,
            'table0:slide1': 0.4,
            'table0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        if no_fence:
            target_range = np.array([[-0.1, 0.1, 0], [0.1, 0.1, 0]])
            file_name = 'fetch/push_no_fence.xml'
        else:
            target_range = 0.01
            file_name = 'fetch/push.xml'
        fetch_env.FetchEnv.__init__(
            self, file_name, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=np.array([0, 0.15, 0]),
            obj_range=0.01, target_range=target_range, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
