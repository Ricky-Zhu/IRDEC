from goal_env.robotics.fetch.push import FetchPushEnv
from goal_env.robotics.fetch.push_ori import FetchPushEnv_Ori

from gym.envs.registration import register

kwargs = {
    'reward_type': 'sparse',
    'no_fence': True
}

# modified env
register(
    id='FetchPush-v2',
    entry_point='goal_env.robotics:FetchPushEnv',
    kwargs=kwargs,
    max_episode_steps=50,
)

# original env
register(
    id='FetchPush-v3',
    entry_point='goal_env.robotics:FetchPushEnv_Ori',
    kwargs={
    'reward_type': 'sparse',
},
    max_episode_steps=50,
)
