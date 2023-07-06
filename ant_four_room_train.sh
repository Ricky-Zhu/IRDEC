python trainer.py --device cuda --seed 10 --start-train --dataset-path /envs_data/ant_four_rooms/fourroom_data_3364.pkl \
--env-name AntMazeTest-v2 --iterations 10000 --n-steps 5 --gamma 0.99 --rollout-steps 1000 --batch-size 256 \
--q-combinator min --init_num_n_step_trajs 5000 --random-start 0 --use-behavior-clone-loss --bc-clone-loss-coef 1e-2 \
--exploration-q-coef 1.0 --curiosity-reward-scaling 16.0 --impact-reward-scaling 0.0 --example-control-coef 1.0 \
--save-model-interval 10000