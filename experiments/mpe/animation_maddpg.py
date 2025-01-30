import sys
import os
from os.path import abspath, dirname

root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)
import time

import numpy as np
from functional_critic import utils

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
import argparse

def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ma_type",
                        type=str,
                        help="Experiment Type",
                        required=True)

    parser.add_argument("--scenario",
                        type=str,
                        help="Scenario Name",
                        required=True)

    parser.add_argument("--seed", default=0,
                        type=int)
    parser.add_argument("--nb_test", default=10,
                        type=int)

    parser.add_argument("--step",
                        type=int, required=True)

    parser.add_argument("--path_to_model",
                        type=str,
                        help="Model Path",
                        required=True)

    parser.add_argument("--render", action='store_true')

    args, unknown = parser.parse_known_args()

    hyper_param_directory = '/hyperparams'
    hyper_parameters_name = f'gym__seed_{args.seed}'
    params = utils.get_hyper_parameters(hyper_parameters_name,
                                        args.path_to_model + hyper_param_directory)
    params['is_gym'] = False
    params['model_filepath'] = f'{args.path_to_model}/logs/seed_{args.seed}/steps_{args.step}'
    params['render'] = args.render
    params['nb_test'] = args.nb_test
    params['seed'] = args.seed
    params['step'] = args.step
    params['scenario'] = args.scenario

    return params


def evaluate_model(env, trainers, seed, max_episode_len=50, kk=True, t=10, log=1000):
    evaluation_rewards = []
    agents_done = []
    agents_collision_count = []
    # kk = True
    for i in range(t):
        evaluation_reward = np.zeros(len(trainers))
        s_eval, done_eval = env.reset(), False
        steps_eval = 0
        while not done_eval:
            action_eval_n = [agent.action(obs) for agent, obs in zip(trainers, s_eval)]

            s_eval_, r_eval, done_eval, _ = env.step(action_eval_n)
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                env.render()
                time.sleep(0.05)

            steps_eval += 1
            done_eval = False
            if max_episode_len and steps_eval == max_episode_len:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)
        d = []
        for ii in env.agents:
            d.append(ii.is_done)
        d.append(all(d))
        agents_done.append(d)

        collision_count = []
        for ii in env.agents:
            collision_count.append(ii.collision_count)
        collision_count.append(sum(collision_count) / len(collision_count))
        agents_collision_count.append(collision_count)

        if i % log == 0:
            print(f'{i + 1} Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, '
                  f'Success Rate: {1.0 * np.array(agents_done)[:, -1].sum() / (i + 1) : .2f}, '
                  f'Collision Avg: {1.0 * np.array(agents_collision_count)[:, -1].sum() / (i + 1) : .2f}')

    print(f'Seed: {seed} - Success Rate: {np.array(agents_done)[:, -1].mean() : .3f} ')
    print(f'Seed: {seed} - Collision Avg: {np.array(agents_collision_count)[:, -1].mean() : .3f}')
    print(f'Seed: {seed} - Team Reward Avg: {np.sum(np.array(evaluation_rewards), axis=1).mean(): .3f}')

    return np.array(agents_done)[:, -1].mean(), \
           np.array(agents_collision_count)[:, -1].mean(), \
           np.sum(np.array(evaluation_rewards), axis=1).mean()


def evaluate_model_catcher(env, trainers, seed, max_episode_len=50, kk=False, t=10, log=1000):
    evaluation_rewards = []
    agents_total_catch = []
    agents_catch_with_three = []
    agents_catch_with_two = []
    agents_catch_with_one = []
    # kk = True
    for i in range(t):
        evaluation_reward = np.zeros(len(trainers))
        s_eval, done_eval = env.reset(), False
        steps_eval = 0
        while not done_eval:
            action_eval_n = [agent.action(obs) for agent, obs in zip(trainers, s_eval)]

            s_eval_, r_eval, done_eval, _ = env.step(action_eval_n)
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                env.render()
                time.sleep(0.05)

            steps_eval += 1
            done_eval = False
            if max_episode_len and steps_eval == max_episode_len:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)


        threes = 0
        twos = 0
        ones = 0
        for ii in env.world.agents[-1].catchers:
            if len(ii) == 3:
                threes += 1
            elif len(ii) == 2:
                twos += 1
            elif len(ii) == 1:
                ones += 1
        agents_catch_with_three.append(threes)
        agents_catch_with_two.append(twos)
        agents_catch_with_one.append(ones)
        agents_total_catch.append(threes + twos + ones)
        if i % log == 0:
            print(f'{i + 1} Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, '
                  f'Total catch average per episode: {np.array(agents_total_catch).mean() : .2f}, '
                  f'Average catch with three agents: {np.array(agents_catch_with_three).mean() : .2f}, '
                  f'Average catch with two agents: {np.array(agents_catch_with_two).mean() : .2f}, '
                  f'Average catch with one agents: {np.array(agents_catch_with_one).mean() : .2f}')

    print(f'Seed: {seed} - Total catch average per episode: {np.array(agents_total_catch).mean() : .2f}, ')
    print(f'Seed: {seed} - Average catch with three agents: {np.array(agents_catch_with_three).mean() : .2f}, ')
    print(f'Seed: {seed} - Average catch with two agents: {np.array(agents_catch_with_two).mean() : .2f}, ')
    print(f'Seed: {seed} - Average catch with one agents: {np.array(agents_catch_with_one).mean() : .2f}, ')
    print(f'Seed: {seed} - Team Reward Avg: {np.sum(np.array(evaluation_rewards), axis=1).mean(): .3f}')

    return np.array(agents_total_catch).mean(), \
           np.array(agents_catch_with_three).mean(), \
           np.array(agents_catch_with_two).mean(), \
           np.array(agents_catch_with_one).mean(), \
           np.sum(np.array(evaluation_rewards), axis=1).mean()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def agent_networks_maddpg(train_env, params):
    obs_shape_n = [train_env.observation_space[i].shape for i in range(train_env.n)]
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(train_env.n):
        trainers.append(trainer(
            f"seed_{_params['seed']}-agent_%d" % i, model, obs_shape_n, train_env.action_space, i, params))
    return trainers


if __name__ == "__main__":
    _params = argument_parser()


    with U.single_threaded_session():
        scenario = scenarios.load(_params["env_name"] + ".py").Scenario()
        # create world
        world = scenario.make_world()
        # create multiagent environment
        _env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                             scenario.observation, width=700, height=700,
                             )
        _trainers = agent_networks_maddpg(_env, _params)
        print('Loading previous state...')
        U.load_state(_params['model_filepath'])
        env_catcher = 'predator' in _params['scenario']
        if not env_catcher:
            success, collision, reward = evaluate_model(_env, _trainers, _params['seed'], 50,
                                                        kk=_params['render'], t=_params['nb_test'])
        else:
            total_catch_per_episode, \
            three_agents_catch_per_episode, \
            two_agents_catch_per_episode, \
            one_agents_catch_per_episode, \
            reward = evaluate_model_catcher(_env, _trainers, _params['seed'], 50,
                                            kk=_params['render'], t=_params['nb_test'])



