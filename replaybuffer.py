import numpy as np
from collections import deque
import random
import pickle
import gym


class N_step_traj:
    """
    a container used to store n steps sub-trajs. can return n-steps states, actions, rewards and final state
    which will be given if done or reach the end of the n-steps traj
    """

    def __init__(self, n_steps=10):
        self.n_steps = n_steps
        self.reset()

    @property
    def length(self):
        return len(self.states)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.final_state = None

    def add(self, state, action, reward, final_sate, done):
        if done:
            return True
        else:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
            flag = self.complete(done)
            if flag:
                self.final_state = final_sate
            return flag

    def dump(self):
        self.states = np.asarray(self.states)
        self.actions = np.asarray(self.actions)
        self.rewards = np.asarray(self.rewards)
        self.dones = np.asarray(self.dones)
        self.final_state = np.asarray(self.final_state)

        return self.states, self.actions, self.rewards, self.dones, self.final_state

    def complete(self, done):
        flag = done or self.length == self.n_steps
        return flag


class UniformReplayBuffer:
    def __init__(self,
                 max_size=int(1e6),
                 n_step=5,
                 ):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.n_step = n_step

    def add_n_step_traj(self, traj):
        self.buffer.append(traj)

    def sample(self, batch_size):
        batch_n_steps_traj = random.sample(self.buffer, batch_size)
        return np.array(batch_n_steps_traj)

    def output_explored_area(self, sample_points_num=1000):
        batches_num = sample_points_num // self.n_step
        states = self.sample(batches_num)[:, 0]
        states = np.concatenate(states)  # turn to sample_num * state dim
        xy_pos = states[:, :2]
        return xy_pos


class ExpertReplayBuffer:
    def __init__(self, env, example_num=200, terminal_offset=50, from_d4rl=True):
        if from_d4rl:
            dataset = env.get_dataset()
            terminals = np.where(dataset['terminals'])[0]
            expert_obs = np.concatenate(
                [dataset['observations'][t - terminal_offset:t] for t in terminals],
                axis=0)
            expert_actions = np.concatenate(
                [dataset['actions'][t - terminal_offset:t] for t in terminals],
                axis=0)
            indices = np.random.choice(
                len(expert_obs), size=example_num, replace=False)
            self.expert_obs = expert_obs[indices]
            self.expert_actions = expert_actions[indices]
            self.index = np.arange(example_num)
        else:
            dataset = env.get_dataset(num_obs=500)
            indices = np.random.choice(
                dataset['observations'].shape[0], size=example_num, replace=False)
            self.expert_obs = dataset['observations'][indices]
            self.expert_actions = dataset['actions'][indices]
            self.index = np.arange(example_num)

    def sample(self, batch_size):
        temp_ind = np.random.choice(self.index, batch_size, replace=False)
        batch_obs = self.expert_obs[temp_ind]
        batch_actions = self.expert_actions[temp_ind]
        return batch_obs, batch_actions

    @property
    def buffer_size(self):
        return self.expert_obs.shape[0]


class ExpertBufferRoboverse:
    def __init__(self, path, num_expert_obs=200, offset=5, near_start=True, pick_and_place=False):
        data_file = open(path, 'rb')
        trajs = pickle.load(data_file)
        data_file.close()

        # select the pairs near the start
        if not pick_and_place:
            if near_start:
                terminals = np.asarray(trajs[2])
                start_points = np.where(terminals == 1)[0]
                start_points = start_points[:-1]
                start_points += 1
                start_points = np.concatenate([[0], start_points], axis=0)  # add the data first point

                # which data to use
                obs = np.concatenate([np.asarray(trajs[0])[t:t + offset] for t in start_points], axis=0)
                actions = np.concatenate([np.asarray(trajs[1])[t:t + offset] for t in start_points], axis=0)
                index = np.random.choice(len(obs), size=num_expert_obs, replace=False)
            else:
                terminals = np.asarray(trajs[2])
                terminals = np.where(terminals == 1)[0]
                obs = np.concatenate(
                    [np.asarray(trajs[0])[t - offset:t] for t in terminals],
                    axis=0)
                actions = np.concatenate(
                    [np.asarray(trajs[1])[t - offset:t] for t in terminals],
                    axis=0)
                index = np.random.choice(
                    len(obs), size=num_expert_obs, replace=False)

            self.obs = obs[index]
            self.actions = actions[index]
        else:
            self.obs = np.asarray(trajs[0])
            self.actions = np.asarray(trajs[1])

        assert self.obs.shape[0] == self.actions.shape[0]

        self.index = np.arange(self.actions.shape[0])
        self.reset()
        self.pointer = 0

    def sample(self, batch_size):
        if self.pointer + batch_size > self.buffer_size:
            self.reset()
        temp_ind = self.index[self.pointer:self.pointer + batch_size]
        self.pointer += batch_size

        return self.obs[temp_ind], self.actions[temp_ind]

    def reset(self):
        np.random.shuffle(self.index)
        self.pointer = 0

    @property
    def buffer_size(self):
        return self.actions.shape[0]

    @property
    def obs_action_dim(self):
        return self.obs.shape[1], self.actions.shape[1]


class ExpertBufferFromPath:
    def __init__(self, path):
        data_file = open(path, 'rb')
        trajs = pickle.load(data_file)
        data_file.close()
        self.obs = np.asarray(trajs[0])
        self.actions = np.asarray(trajs[1])

        assert self.obs.shape[0] == self.actions.shape[0]

        self.index = np.arange(self.actions.shape[0])
        self.reset()
        self.pointer = 0

    def sample(self, batch_size):
        if self.pointer + batch_size > self.buffer_size:
            self.reset()
        temp_ind = self.index[self.pointer:self.pointer + batch_size]
        self.pointer += batch_size

        return self.obs[temp_ind], self.actions[temp_ind]

    def reset(self):
        np.random.shuffle(self.index)
        self.pointer = 0

    @property
    def buffer_size(self):
        return self.actions.shape[0]

    @property
    def obs_action_dim(self):
        return self.obs.shape[1], self.actions.shape[1]
