python trainer.py --device cuda --seed 10 --start-train --dataset-path /envs_data/ant_maze/expert_traj_3688.pkl \
--env-name AntMaze1Test-v1 --iterations 4100 --n-steps 5 --gamma 0.99 --evaluation-rollouts 10 --rollout-steps 500 \
--batch-size 256 --q-combinator min --init_num_n_step_trajs 5000 --random-start 0 --use-behavior-clone-loss \
--bc-clone-loss-coef 1e-2 --exploration-q-coef 1.0 --curiosity-reward-scaling 1.0 --impact-reward-scaling 0.01 \
--example-control-coef 1.0 --save-model-interval 10000