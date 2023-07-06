from env_utils import load_env_and_expert_buffer, get_env_params
from argments import get_args
from agent import RceAgent
import torch
import numpy as np
import random

if __name__ == "__main__":
    def set_seeds(args, rank=0):
        # set seeds for the numpy
        np.random.seed(args.seed + rank)
        # set seeds for the random.random
        random.seed(args.seed + rank)
        # set seeds for the pytorch
        torch.manual_seed(args.seed + rank)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed + rank)
            torch.cuda.manual_seed_all(args.seed + rank)


    args = get_args()

    set_seeds(args)
    env, expert_buffer = load_env_and_expert_buffer(args)
    env_params = get_env_params(env)

    agent = RceAgent(env, args, env_params, expert_examples_buffer=expert_buffer, load_previous_model_path=None)

    agent.learn()
