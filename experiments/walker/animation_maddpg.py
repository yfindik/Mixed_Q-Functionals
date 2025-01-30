import sys
import os
from os.path import abspath, dirname
root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)

import argparse
import numpy as np
import torch
import sys

from pettingzoo.sisl import multiwalker_v9

from functional_critic import utils, utils_for_q_learning

from walker_utils_ma import multiwalker_reset_separate, multiwalker_step_separate, \
    multiwalker_adjustment_separate, multiwalker_adjustment_centralized, \
    multiwalker_step_centralized, multiwalker_reset_centralized
import tensorflow as tf
import tensorflow.contrib.layers as layers
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg_modified import MADDPGAgentTrainer


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ma_type",
                        type=str,
                        help="Experiment Type",
                        required=True)

    parser.add_argument("--scenario",
                        type=str,
                        help="Scenario Name",
                        required=False)

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

    return params


def evaluate_model(env, trainers, t=100, log=1000):
    evaluation_rewards = []
    fail_count = 0
    kk = False
    for i in range(t):
        evaluation_reward = np.zeros(env.n)
        s_eval, done_eval = multiwalker_reset_separate(env), False
        steps_eval = 0
        max_r = 0
        min_r = 10
        while not done_eval:
            action_eval_n = {}
            for agent, agent_name in zip(trainers, env.agents):
                action_eval_n[agent_name] = list(np.clip(agent.action(s_eval[agent_name]), -1, 1))

            # environment step
            try:
                data_eval = multiwalker_step_separate(env, action_eval_n)
            except Exception as e:
                print(e)

            s_eval_, r_eval, done_eval = data_eval[0], data_eval[1], data_eval[2]

            try:
                evaluation_reward += np.array(list(r_eval.values())[:env.n])
            except Exception as e:
                print(e)

            # evaluation_reward += r_eval
            s_eval = s_eval_
            if list(r_eval.values())[0] > max_r:
                max_r = list(r_eval.values())[0]

            if list(r_eval.values())[0] < min_r:
                min_r = list(r_eval.values())[0]

            if kk:
                env.render()

            steps_eval += 1
            done_eval = all(done_eval.values())

        if min_r < -50:
            fail_count += 1
        if i % log == 0:
            print(f'{i} Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, Minimum Reward: {min_r}')
        evaluation_rewards.append(evaluation_reward)

    print(f'fail_count: {fail_count}')

    # there is a bug for 2 walker with averagin withouth using axis dim
    return np.array(evaluation_rewards).mean(), 1 - (fail_count/t)


def mlp_model_actor_maddpg(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.tanh)
        return out


def mlp_model_critic_maddpg(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_walker(params, adj=False, centralized=False):

    train_env = multiwalker_v9.parallel_env(n_walkers=params['n_walkers'],
                                            shared_reward=False,
                                            terrain_length=params['terrain_length'],
                                            max_cycles=params['max_episode_len'],
                                            position_noise=0, angle_noise=0,
                                            forward_reward=params['forward_reward'])
    eval_env = multiwalker_v9.parallel_env(n_walkers=params['n_walkers'],
                                           shared_reward=False,
                                           terrain_length=params['terrain_length'],
                                           max_cycles=params['max_episode_len'],
                                           position_noise=0, angle_noise=0,
                                           forward_reward=params['forward_reward'])
    if adj:
        if centralized:
            train_env = multiwalker_adjustment_centralized(train_env, obs_indexes=params['obs_indexes'])
            eval_env = multiwalker_adjustment_centralized(eval_env, obs_indexes=params['obs_indexes'])
        else:
            train_env = multiwalker_adjustment_separate(train_env, obs_indexes=params['obs_indexes'])
            eval_env = multiwalker_adjustment_separate(eval_env, obs_indexes=params['obs_indexes'])

    params['env'] = train_env
    utils_for_q_learning.set_random_seed(params)

    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string
    params["nb_agents"] = len(train_env.action_spaces)

    utils.save_hyper_parameters(params, params['seed'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("running on CUDA...")
    else:
        device = torch.device("cpu")
        print("running on CPU...")

    return train_env, eval_env, device, params


def agent_networks_maddpg(train_env, params):
    obs_shape_n = [list(train_env.observation_spaces.values())[i].shape for i in range(train_env.n)]
    trainers = []
    model_actor = mlp_model_actor_maddpg
    model_critic = mlp_model_critic_maddpg
    trainer = MADDPGAgentTrainer
    for i in range(train_env.n):
        trainers.append(trainer(
            f"seed_{_params['seed']}-agent_%d" % i, model_actor,
            model_critic, obs_shape_n,
            list(train_env.action_spaces.values()), i, params))
    return trainers


if __name__ == "__main__":
    _params = argument_parser()


    _params['obs_indexes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30]
    with U.single_threaded_session():
        if _params['render']:
            _env = multiwalker_v9.parallel_env(n_walkers=_params['n_walkers'], shared_reward=False,
                                               terrain_length=30, max_cycles=500,
                                               position_noise=0, angle_noise=0,
                                               forward_reward=5.0,
                                               render_mode='human'
                                               )
        else:
            _env = multiwalker_v9.parallel_env(n_walkers=_params['n_walkers'], shared_reward=False,
                                               terrain_length=30, max_cycles=500,
                                               position_noise=0, angle_noise=0,
                                               forward_reward=5.0
                                               )

        _env = multiwalker_adjustment_separate(_env, obs_indexes=_params['obs_indexes'])
        _trainers = agent_networks_maddpg(_env, _params)

        print('Loading previous state...')
        U.load_state(_params['model_filepath'])
        reward, success = evaluate_model(_env, _trainers, _params['nb_test'])
        U.reset()

    print(f'ma_type: {_params["ma_type"]}, '
          f'seed: {_params["seed"]}, '
          f'evaluation reward: {reward}, '
          f'success: {success}')


