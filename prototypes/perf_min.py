import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from benchutils.chrono import show_eta, MultiStageChrono, time_this

from a2c_ppo_acktr.distributions import Categorical, DiagGaussian, Bernoulli
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


chrono = MultiStageChrono()


class CNNBase(nn.Module):
    def __init__(self, num_inputs, hidden_size=512):
        super(CNNBase).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x, rnn_hxs


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.base = CNNBase(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)

        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)

        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class ProximalPolicyOptimization:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 chrono=chrono):

        self.actor_critic = actor_critic
        self.chrono = chrono

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def compute_cost(self, action_log_probs, old_action_log_probs_batch, adv_targ, value_preds_batch, values, return_batch, dist_entropy, **kwargs):
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(- self.clip_param,  self.clip_param)
        value_losses = (values - return_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
        return loss, value_loss, action_loss

    @time_this(chrono, verbose=True)
    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        #
        with self.chrono.time('ppo_epoch', verbose=True):
            for e in range(self.ppo_epoch):     # 4

                with self.chrono.time('generate_data', verbose=True):
                    if self.actor_critic.is_recurrent:
                        data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
                    else:
                        data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

                # this can be parallelized
                with self.chrono.time('samples', verbose=True):
                    for sample in data_generator: # 32
                        with self.chrono.time('sample', verbose=True):
                            obs_batch, recurrent_hidden_states_batch, actions_batch, \
                               value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ = sample

                            # Reshape to do in a single forward pass for all steps
                            with self.chrono.time('evaluate', verbose=True):
                                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                                    obs_batch,
                                    recurrent_hidden_states_batch,
                                    masks_batch,
                                    actions_batch)

                            self.optimizer.zero_grad()

                            with self.chrono.time('cost', verbose=True):
                                loss, value_loss, action_loss = ProximalPolicyOptimization.compute_cost(**locals())

                            with self.chrono.time('optimizer_step', verbose=True):
                                loss.backward()

                                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                                self.optimizer.step()

                            value_loss_epoch += value_loss.item()
                            action_loss_epoch += action_loss.item()
                            dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


def get_args():

    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--repeat', type=int, default=100, help='number of observation timed')
    parser.add_argument('--number', type=int, default=10, help='number of time a task is done in between timer')
    parser.add_argument('--report', type=str, default=None, help='file to store the benchmark result in')
    parser.add_argument('--seed', default=1)

    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')

    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=10e6,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')

    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')

    return parser


args = get_args().parse_args()


def make_empty_dir(name):
    import glob
    import os

    try:
        os.makedirs(name)
    except OSError:
        files = glob.glob(os.path.join(name, '*.monitor.csv'))
        for f in files:
            os.remove(f)


make_empty_dir(args.log_dir)


def generate_rollouts(rollouts, actor_critic, envs, episode_rewards, **kwargs):

    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step],
                rollouts.recurrent_hidden_states[step],
                rollouts.masks[step]
            )

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)


def main():
    import copy
    import glob
    import os
    import time
    from collections import deque

    import gym
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from a2c_ppo_acktr import algo
    from a2c_ppo_acktr.envs import make_vec_envs
    from a2c_ppo_acktr.storage import RolloutStorage
    from a2c_ppo_acktr.utils import get_vec_normalize, update_linear_schedule
    from a2c_ppo_acktr.visualize import visdom_plot

    device = torch.device('cuda')

    envs = make_vec_envs(
        args.env_name,
        args.seed,
        args.num_processes,
        args.gamma,
        args.log_dir,
        args.add_timestep,
        device,
        False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)

    actor_critic.to(device)

    agent = ProximalPolicyOptimization(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        chrono=chrono
    )

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    start = time.time()
    for j in range(args.repeat):
        with chrono.time('train', verbose=True) as t:
            for n in range(args.number):

                with chrono.time('one_batch', verbose=True):

                    if args.use_linear_lr_decay:
                        # decrease learning rate linearly
                        if args.algo == "acktr":
                            # use optimizer's learning rate since it's hard-coded in kfac.py
                            update_linear_schedule(agent.optimizer, j, num_updates, agent.optimizer.lr)
                        else:
                            update_linear_schedule(agent.optimizer, j, num_updates, args.lr)

                    if args.algo == 'ppo' and args.use_linear_clip_decay:
                        agent.clip_param = args.clip_param * (1 - j / float(num_updates))

                    with chrono.time('generate_rollouts', verbose=True):
                        generate_rollouts(**locals())

                        with torch.no_grad():
                            next_value = actor_critic.get_value(
                                rollouts.obs[-1],
                                rollouts.recurrent_hidden_states[-1],
                                rollouts.masks[-1]
                            ).detach()

                    # ---
                    with chrono.time('compute_returns', verbose=True):
                        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

                    with chrono.time('agent.update', verbose=True):  # 11.147009023304644
                        value_loss, action_loss, dist_entropy = agent.update(rollouts)

                        #exp.log_batch_loss(action_loss)
                        #exp.log_metric('value_loss', value_loss)

                    with chrono.time('after_update', verbose=True):
                        rollouts.after_update()

                    total_num_steps = (j + 1) * args.num_processes * args.num_steps

                    #if args.eval_interval is not None and len(episode_rewards) > 1 and j % args.eval_interval == 0:
                    #    eval_model(**locals())

            # -- number
        # -- chrono
        #exp.show_eta(j, t)

    # -- epoch
    #exp.report()

if __name__ == "__main__":
    main()
