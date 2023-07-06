import gym
from goal_env.mujoco import *
from replaybuffer import ExpertBufferFromPath
from gym.wrappers import TimeLimit
from wrappers import StepWrapperAntMaze
import os


def load_env_and_expert_buffer(args, return_buffer=True):
    expert_buffer = None
    data_path = os.path.abspath(
        os.path.dirname(__file__)) + args.dataset_path

    env = gym.make(args.env_name, random_start=args.random_start)
    env.seed(args.seed)
    env = StepWrapperAntMaze(env)
    env = TimeLimit(env, max_episode_steps=args.rollout_steps + 1)
    if return_buffer:
        expert_buffer = ExpertBufferFromPath(path=data_path)

    return env, expert_buffer


def get_env_params(env):
    s = env.reset()
    if isinstance(s, dict):
        s = s['observation']
    env_params = {'max_action': env.action_space.high[0],
                  'state_dim': s.shape[0],
                  'action_dim': env.action_space.shape[0]}
    return env_params
