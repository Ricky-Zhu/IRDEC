import torch
import numpy as np
from replaybuffer import UniformReplayBuffer, N_step_traj
from models import tanh_gaussian_actor, Critic, ICMModel
from torch.distributions.normal import Normal
import copy
import imageio
import os
from datetime import datetime
import wandb
from tqdm import tqdm
from logger import PathLogger
from torch.nn import functional as F


class RceAgent:
    def __init__(self, env, args, env_params, expert_examples_buffer=None, load_previous_model_path=None):

        if args.start_train:
            current_date = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
            self.model_save_path = os.path.abspath(os.path.dirname(__file__)) + '/saved_models/{}_{}'.format(
                args.env_name, current_date)
            if not os.path.exists(self.model_save_path):
                os.makedirs(self.model_save_path)

        if args.start_train:
            wandb.login()
            wandb.init(
                project="rce_pytorch_{}".format(args.env_name),
                config=vars(args),
                name="rce_{}_{}".format(args.env_name, current_date)
            )

        self.env = env
        self.args = wandb.config if args.start_train else args
        self.env_params = env_params
        self.device = self.args.device
        self.max_action = self.env_params['max_action']
        self.n_steps_traj_rec = N_step_traj(n_steps=self.args.n_steps)
        self.replay_buffer = UniformReplayBuffer(max_size=self.args.buffer_size, n_step=self.args.n_steps)
        self.path_logger = PathLogger(episode_length=self.args.rollout_steps)

        self.expert_buffer = expert_examples_buffer
        self.critic_criterion = torch.nn.MSELoss(reduction='none')

        self.actor = tanh_gaussian_actor(self.env_params['state_dim'], self.env_params['action_dim'],
                                         self.args.hidden_size
                                         ).to(self.device)

        self.critic_1 = Critic(self.env_params['state_dim'], self.env_params['action_dim'],
                               self.args.hidden_size, loss_type='c').to(self.device)
        self.critic_2 = Critic(self.env_params['state_dim'], self.env_params['action_dim'],
                               self.args.hidden_size, loss_type='c').to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)

        # optimizers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(),
                                               lr=self.args.critic_lr,
                                               weight_decay=1e-5)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(),
                                               lr=self.args.critic_lr,
                                               weight_decay=1e-5)

        ############ setup the icm model and the exploration value head ####################
        self.icm = ICMModel(input_size=self.env_params['state_dim'],
                            output_size=self.env_params['action_dim'],
                            use_action_embedding=True, device=self.device).to(self.device)
        self.icm_optim = torch.optim.Adam(self.icm.parameters(), lr=3e-5)

        self.explore_q = Critic(self.env_params['state_dim'],
                                self.env_params['action_dim'],
                                self.args.hidden_size).to(self.device)
        self.target_explore_q = copy.deepcopy(self.explore_q).to(self.device)
        self.explore_q_optim = torch.optim.Adam(self.explore_q.parameters(), lr=self.args.critic_lr)

        #####################################################
        # for normalizing the impact bonus across different tasks
        self.impact_bonus_avg = torch.tensor(0.0).to(self.device)
        self.impact_n = torch.tensor(0).int().to(self.device)

        self.global_step = 0
        if load_previous_model_path is not None:
            # load the actor, critic1, 2 and replay buffer
            abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'saved_models',
                                    load_previous_model_path)
            actor_path = os.path.join(abs_path, 'actor_model_1000000.pt')
            critic_path = os.path.join(abs_path, 'critic_model.pt')
            buffer_path = os.path.join(abs_path, 'replay_buffer.pt')

            actor_check = torch.load(actor_path)
            self.actor.load_state_dict(actor_check)

            critic_check = torch.load(critic_path)
            self.critic_1.load_state_dict(critic_check['q1_model'])
            self.target_critic_1.load_state_dict(critic_check['q1_model'])
            self.critic_2.load_state_dict(critic_check['q2_model'])
            self.target_critic_2.load_state_dict(critic_check['q2_model'])

            self.replay_buffer = torch.load(buffer_path)

            self.global_step = 1000000

    def update(self):
        expert_states, expert_actions_from_buffer = self.expert_buffer.sample(self.args.batch_size)
        expert_states = self.to_tensor(expert_states)  # success examples
        expert_actions_from_buffer /= self.max_action
        expert_actions_from_buffer = self.to_tensor(expert_actions_from_buffer)

        transitions = self.replay_buffer.sample(self.args.batch_size)
        processed_batches = [self._preprocess_n_steps_traj_batches(x) for x in transitions]

        # s_{t},a_{t},s_{t+1},s_{t+n},d_{t+n}
        states, actions, rewards_ext, next_states, future_states, dones = map(np.array, zip(*processed_batches))
        states = self.to_tensor(states)
        actions = self.to_tensor(actions)
        next_states = self.to_tensor(next_states)
        future_states = self.to_tensor(future_states)
        rewards_ext = self.to_tensor(rewards_ext)
        dones = self.to_tensor(dones[:, 0])

        ############## obtain the intrinsic reward and train the icm module ################
        real_next_state_feature, pred_next_state_feature, pred_action, action_embed, impact_diff = self.icm(
            (states, next_states, actions))
        idm_loss = F.mse_loss(pred_action, action_embed).mean()
        fdm_loss = F.mse_loss(pred_next_state_feature, real_next_state_feature, reduction='none').mean(-1)

        # update the running avg of the impact diff
        self.impact_bonus_avg = (self.impact_n * self.impact_bonus_avg + torch.sum(impact_diff)) / (
                self.impact_n + impact_diff.shape[0])
        self.impact_n += impact_diff.shape[0]

        curiosity_rew = torch.clone(fdm_loss).detach()

        impact_rew = 1 - 1 / (impact_diff + 1.0)

        rewards_icm = self.args.curiosity_reward_scaling * curiosity_rew + self.args.impact_reward_scaling * impact_rew
        rewards = rewards_icm + rewards_ext

        fdm_loss = fdm_loss.mean()

        icm_loss = self.args.idm_coef * idm_loss + self.args.fdm_coef * fdm_loss  ### adjust the coefficients
        self.icm_optim.zero_grad()
        icm_loss.backward()
        self.icm_optim.step()

        # update the exploration bonus value head
        with torch.no_grad():
            action_next, _ = self.select_action(next_states)
            exploration_q_target = rewards[:, None] + self.args.gamma * (
                    1. - dones[:, None].float()) * self.target_explore_q(next_states,
                                                                         action_next)

        pred_exploration_q = self.explore_q(states, actions)
        exploration_q_loss = F.mse_loss(pred_exploration_q, exploration_q_target).mean()
        self.explore_q_optim.zero_grad()
        exploration_q_loss.backward()
        self.explore_q_optim.step()

        self._soft_update_target_network(self.target_explore_q, self.explore_q)

        ###################################################################################

        # compute the targets
        with torch.no_grad():
            next_actions, _ = self.select_action(next_states)
            target_q_1 = self.target_critic_1(next_states, next_actions)
            target_q_2 = self.target_critic_2(next_states, next_actions)

            future_actions, _ = self.select_action(future_states)
            target_q_future_1 = self.target_critic_1(future_states, future_actions)
            target_q_future_2 = self.target_critic_2(future_states, future_actions)

            gamma_n = self.args.gamma ** self.args.n_steps
            target_q_1 = (target_q_1 + gamma_n * target_q_future_1) / 2.0
            target_q_2 = (target_q_2 + gamma_n * target_q_future_2) / 2.0

            if self.args.q_combinator == 'min':
                target_q = torch.min(target_q_1, target_q_2)
            else:
                target_q = torch.max(target_q_1, target_q_2)

            w = target_q / (1. - target_q)
            td_targets = self.args.gamma * w / (1. + self.args.gamma * w)

        td_targets = torch.cat([torch.ones(self.args.batch_size, 1).to(self.device), td_targets], dim=0)
        weights = torch.cat(
            [torch.ones(self.args.batch_size, 1).to(self.device) - self.args.gamma, 1. + self.args.gamma * w],
            dim=0)

        # compute the predictions
        expert_actions, _ = self.select_action(expert_states)
        pred_expert_1 = self.critic_1(expert_states, expert_actions)
        pred_expert_2 = self.critic_2(expert_states, expert_actions)

        pre_exp1_log = pred_expert_1.detach().clone().mean().cpu().numpy()
        pre_exp2_log = pred_expert_2.detach().clone().mean().cpu().numpy()

        pred_1 = self.critic_1(states, actions)
        pred_2 = self.critic_2(states, actions)

        if self.args.start_train:
            wandb.log({'c_value/c_q_1_exp': pre_exp1_log,
                       'c_value/c_q_2_exp': pre_exp2_log,
                       'c_value/c_actor_1': pred_1.detach().clone().mean().cpu().numpy(),
                       'c_value/c_actor_2': pred_2.detach().clone().mean().cpu().numpy()}, self.global_step)

        pred_1 = torch.cat([pred_expert_1, pred_1], dim=0)
        pred_2 = torch.cat([pred_expert_2, pred_2], dim=0)

        critic_1_loss = self.args.critic_loss_coef * (weights * self.critic_criterion(pred_1, td_targets)).mean()
        critic_2_loss = self.args.critic_loss_coef * (weights * self.critic_criterion(pred_2, td_targets)).mean()

        self.critic_1_optim.zero_grad()
        critic_1_loss.backward(retain_graph=True)
        self.critic_1_optim.step()
        self.critic_2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optim.step()

        # soft update the target critic networks
        self._soft_update_target_network(self.target_critic_1, self.critic_1)
        self._soft_update_target_network(self.target_critic_2, self.critic_2)

        # update the actor
        actions_new, _, log_probs = self.select_action(states, return_action_log_probs=True)
        target_q1 = self.critic_1(states, actions_new)
        target_q2 = self.critic_2(states, actions_new)

        if self.args.q_combinator == 'min':
            targets_q = torch.min(target_q1, target_q2)
        else:
            targets_q = torch.max(target_q1, target_q2)

        ### add the exploration q value head ##################
        exploration_q_target = self.explore_q(states, actions_new)
        #######################################################

        # the coefficient needs to be adjusted, turn off the example control to see how the exploration bonus works
        actor_loss = self.args.actor_loss_coef * (
                self.args.entropy_coef * log_probs - self.args.example_control_coef * targets_q - self.args.exploration_q_coef * exploration_q_target).mean()

        if self.args.use_behavior_clone_loss:
            mean, std = self.actor(expert_states)
            expert_actions_log_probs = Normal(mean, std).log_prob(expert_actions_from_buffer).sum(1)
            bc_loss = -expert_actions_log_probs.mean()
            actor_loss += self.args.bc_clone_loss_coef * bc_loss

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item(), idm_loss.item(), exploration_q_loss.item(), fdm_loss.item(), icm_loss.item()

    def collect_rollouts(self):
        num_n_step_traj = 0
        s = self.env.reset()
        for i in range(self.args.rollout_steps):
            action, _ = self.select_action(self.to_tensor(s[None, :]), False)
            action = action.cpu().numpy().squeeze(0)
            s_, r, done, _ = self.env.step(action * self.max_action)
            if_complete = self.n_steps_traj_rec.add(s, action, r, s_, done)

            if if_complete:
                if not done:
                    self.replay_buffer.add_n_step_traj(self.n_steps_traj_rec.dump())
                    num_n_step_traj += 1
                self.n_steps_traj_rec.reset()
            s = s_
            if done:
                s = self.env.reset()
        return num_n_step_traj

    def compute_intrinsic_rew(self, state, next_state, action):
        real_next_state_feature, pred_next_state_feature, pred_action, action = self.icm(
            (state, next_state, action))

        reward = F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
        return reward.data.cpu().numpy()

    def evaluate(self, model_path=None, render_mode='human', animate=None, eval_num=None,
                 rollout_steps=None):
        frame_count = 0
        if_animate = self.args.animate
        rollout_num = eval_num if eval_num is not None else self.args.evaluation_rollouts
        rollout_steps = rollout_steps if rollout_steps is not None else self.args.rollout_steps
        success_count = 0
        first_count = True
        if model_path is not None:
            checkpoint = torch.load(model_path)
            self.actor.load_state_dict(checkpoint)
        if animate is not None:
            if_animate = animate

        self.actor.eval()
        episode_rew = 0

        for i in range(rollout_num):
            s = self.env.reset()
            for _ in range(rollout_steps):
                self.path_logger.log(s)
                action, _ = self.select_action(self.to_tensor(s[None, :]), False)
                action = action.cpu().numpy().squeeze(0)
                s_, r, done, info = self.env.step(action * self.max_action)

                if info['is_success'] and first_count:
                    success_count += 1
                    first_count = False

                if if_animate:
                    frame = self.env.render(render_mode)

                episode_rew += r
                s = s_

                if done:
                    break
            first_count = True
        self.actor.train()
        success_rate = float(success_count) / float(rollout_num)
        average_episode_return = episode_rew / float(rollout_num)

        return average_episode_return, success_rate

    def learn(self):
        print('###### initialize replay buffer ######')
        self.init_replay_buffer()
        print('###### finish initialization ######')
        if self.args.start_train:
            wandb.watch(self.actor, log_freq=1000)
        for _ in tqdm(range(self.args.iterations)):
            num_n_step_traj = self.collect_rollouts()
            for _ in range(num_n_step_traj):
                q1_loss, q2_loss, actor_loss, idm_loss, exploration_q_loss, fdm_loss, icm_loss = self.update()

                if self.args.start_train:
                    wandb.log({'loss/q1_loss': q1_loss,
                               'loss/q2_loss': q2_loss,
                               'loss/actor_loss': actor_loss,
                               'loss/idm_loss': idm_loss,
                               'loss/exploration_q_loss': exploration_q_loss,
                               'loss/fdm_loss': fdm_loss,
                               'loss/icm_loss': icm_loss}, step=self.global_step)

                self.global_step += 1
                if self.global_step % self.args.save_model_interval == 0:
                    eval_rew, eval_success_rate = self.evaluate()
                    print('episode return:{}, evaluation_success_rate:{},global_step:{}'.format(eval_rew,
                                                                                                eval_success_rate,
                                                                                                self.global_step))

                    if self.args.start_train:
                        wandb.log({'episode return': eval_rew}, step=self.global_step)
                        wandb.log({'success rate': eval_success_rate}, step=self.global_step)

                        # save model

                        critic_model_path = os.path.join(self.model_save_path, 'critic_model.pt')
                        actor_model_path = os.path.join(self.model_save_path, 'actor_model_{}.pt'.format(
                            self.global_step))

                        # np.save(os.path.join(self.model_save_path, 'area_{}.npy'.format(self.global_step)),
                        #         self.output_explored_area())
                        torch.save({'q1_model': self.critic_1.state_dict(),
                                    'q2_model': self.critic_2.state_dict()}, critic_model_path)
                        # torch.save(self.replay_buffer, replay_buffer_path)
                        # torch.save(self.path_logger, logger_path)

                        torch.save(self.actor.state_dict(), actor_model_path)

    def _preprocess_n_steps_traj_batches(self, x):

        states, actions, rewards, dones, final_state = x
        state, next_state, future_state = states[0], states[1], final_state
        action = actions[0]

        #### use the cumulative rewards
        r = 0
        for i in reversed(range(len(rewards))):
            r += self.args.gamma * r + rewards[i]

        return state, action, r, next_state, future_state, dones

    def select_action(self, s, rsample=True, return_action_log_probs=False):

        mean, std = self.actor(s)
        std += 1e-3  # for numerical stability

        if rsample:
            pre_tanh_action = mean + torch.randn(mean.size()).to(self.device) * std
            action = torch.tanh(pre_tanh_action)
            action.requires_grad_()
        else:
            pre_tanh_action = Normal(mean, std).sample()
            action = torch.tanh(pre_tanh_action).detach()

        if return_action_log_probs:
            actions_probs = Normal(mean, std).log_prob(pre_tanh_action) - torch.log(1 - action ** 2 + 1e-6)
            actions_probs = actions_probs.sum(dim=1, keepdim=True)
            return action, pre_tanh_action, actions_probs

        return action, pre_tanh_action

    def init_replay_buffer(self):
        init_collected_n_steps_traj = 0
        while init_collected_n_steps_traj < self.args.init_num_n_step_trajs:
            num = self.collect_rollouts()
            init_collected_n_steps_traj += num

    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def to_tensor(self, x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype).to(self.device)

    def output_explored_area(self, sample_points_num=2000):
        xy_pos = self.replay_buffer.output_explored_area(sample_points_num)
        return xy_pos
