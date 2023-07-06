import argparse


def get_args():
    parser = argparse.ArgumentParser(description='IRDEC')
    parser.add_argument('--env-name', type=str, default='AntMaze1Test-v1')
    parser.add_argument('--device', type=str, default='cuda')

    # training hyper general
    parser.add_argument('--n-steps', type=int, default=5)
    parser.add_argument('--buffer-size', type=int, default=int(1e6))
    parser.add_argument('--rollout-steps', type=int, default=1001)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--evaluation-rollouts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--save-model-interval', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=1e-4)
    parser.add_argument('--num-expert-obs', type=int, default=300)
    parser.add_argument('--display-loss-interval', type=int, default=1000)
    parser.add_argument('--init_num_n_step_trajs', type=int, default=100)
    parser.add_argument('--polyak', type=float, default=0.95)
    parser.add_argument('--q-combinator', type=str, default='min')
    parser.add_argument('--use-automatic-entropy-reg', action='store_true')
    parser.add_argument('--critic-loss-coef', type=float, default=0.5)
    parser.add_argument('--actor-loss-coef', type=float, default=1.0)
    parser.add_argument('--use-behavior-clone-loss', action='store_true')
    parser.add_argument('--bc-clone-loss-coef', type=float, default=0.01)
    parser.add_argument('--idm-coef', type=float, default=1.0)
    parser.add_argument('--fdm-coef', type=float, default=1.0)
    parser.add_argument('--curiosity-reward-scaling', type=float, default=1.0)
    parser.add_argument('--impact-reward-scaling', type=float, default=.0)
    parser.add_argument('--exploration-q-coef', type=float, default=0.3)
    parser.add_argument('--example-control-coef', type=float, default=1.0)

    parser.add_argument('--use-sac', action='store_true')

    parser.add_argument('--random-start', type=int, default=0)

    # model
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)

    # the dataset used
    parser.add_argument('--dataset-path', type=str, default='/envs_data/ant_maze/expert_traj_3688.pkl')

    # hierarchical special
    parser.add_argument('--latent-dim', type=int, default=128)

    # general
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--start-train', action='store_true', help='start training otherwise debug mode')

    args = parser.parse_args()
    return args
