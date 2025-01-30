import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ddpg.network import ActorNetwork, CriticNetwork
from ddpg.noise_injector import OrnsteinUhlenbeckActionNoise
from ddpg.replaybuffer import ReplayBuffer


class Agent:

    def __init__(self, input_dims, n_actions, params, device, max_action=None):

        self.gamma = params['gamma']
        self.tau = params['target_network_learning_rate']
        layer1_size = params['layer_size']
        layer2_size = params['layer_size']
        batch_size = params['batch_size']
        max_size = params['max_buffer_size']
        lr_actor = params['learning_rate']
        lr_critic = params['learning_rate'] * 10
        # lr_actor = 0.000025
        # lr_critic = 0.00025
        # lr_actor = 0.00005
        # lr_critic = 0.0005

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(n_actions))
        self.replay_buffer = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(
            lr_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Actor', device=device)
        self.target_actor = ActorNetwork(
            lr_actor, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetActor', device=device)

        self.critic = CriticNetwork(
            lr_critic, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='Critic', device=device)
        self.target_critic = CriticNetwork(
            lr_critic, input_dims, layer1_size, layer2_size, n_actions=n_actions, name='TargetCritic', device=device)

        self.update_network_parameters(tau=1)

        self.batch_size = batch_size
        self.max_action = max_action

        T.manual_seed(params['seed'])
        np.random.seed(params['seed'])

    def select_action(self, observation, noise=True): # noise=False
        self.actor.eval()  # Turn eval mode on. This is just for inference.

        observation = T.tensor(
            observation, dtype=T.float).to(self.actor.device)
        mu = self.actor(observation).to(self.actor.device)
        if noise:

            mu += T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
            if self.max_action:
                mu = mu.clip(-1 * self.max_action, self.max_action)

        self.actor.train()  # Recover model mode
        return mu.cpu().detach().numpy()

    def update(self):

        # If memory is not big enough to train, skip it.
        if self.replay_buffer.total_count < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_minibatch(
            self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        # Freeze following network
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(next_state)
        critic_value_next = self.target_critic.forward(
            next_state, target_actions)
        # previous value, to be updated.
        critic_value = self.critic.forward(state, action)

        critic_target = []
        for j in range(self.batch_size):
            critic_target.append(
                reward[j] + self.gamma * critic_value_next[j] * done[j])

        critic_target = T.tensor(critic_target).to(self.critic.device)
        critic_target = critic_target.view(self.batch_size, 1)

        # now update critic network
        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(critic_target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        self.critic.eval()  # now, freeze it for actor update

        # now update actor network
        self.actor.train()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        # output of critic is Q value, which needs to be maximized. Therefore put negative sign to this to minimize.
        actor_q = self.critic.forward(state, mu)
        actor_loss = T.mean(-actor_q)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters(self.tau)

    def update_network_parameters(self, tau):

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1-tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1-tau) * target_actor_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save(self, _dir, debug=False):
        self.actor.save_checkpoint(_dir, debug)
        self.critic.save_checkpoint(_dir, debug)
        self.target_actor.save_checkpoint(_dir, debug)
        self.target_critic.save_checkpoint(_dir, debug)

    def load(self, _dir, debug=False):
        self.actor.load_checkpoint(_dir, debug)
        self.critic.load_checkpoint(_dir, debug)
        self.target_actor.load_checkpoint(_dir, debug)
        self.target_critic.load_checkpoint(_dir, debug)
