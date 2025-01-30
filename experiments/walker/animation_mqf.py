import sys
import os
from os.path import abspath, dirname
root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)
import argparse
import numpy as np
import torch


from pettingzoo.sisl import multiwalker_v9

from functional_critic import utils, utils_for_q_learning
from functional_critic.agents_for_walkers import FourierAgent, LegendreAgent, PolynomialAgent


from walker_utils_ma import multiwalker_reset_separate, multiwalker_step_separate, \
    multiwalker_adjustment_separate, multiwalker_adjustment_centralized, \
    multiwalker_step_centralized, multiwalker_reset_centralized


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
    params['model_filepath'] = f'{args.path_to_model}/logs/seed_{args.seed}_object_x_steps_{args.step}'
    params['render'] = args.render
    params['nb_test'] = args.nb_test
    params['seed'] = args.seed
    params['step'] = args.step

    return params



def agent_networks(params, train_env, device):
    """
    Define online and offline networks
    :param params:
    :param train_env:
    :param device:
    :return:
    """

    Q_Constructor = None

    assert params['functional'] in ["fourier", "polynomial", "legendre"], "Functional type is not acceptable!"
    if params['functional'] == "fourier":
        Q_Constructor = FourierAgent
    elif params['functional'] == "polynomial":
        Q_Constructor = PolynomialAgent
    elif params['functional'] == "legendre":
        Q_Constructor = LegendreAgent

    Q_objects = []
    Q_targets = []

    for _ in range(params['nb_agents']):

        Q_object = Q_Constructor(
            params,
            train_env,
            device=device,
            seed=params['seed']
        )
        Q_target = Q_Constructor(
            params,
            train_env,
            device=device,
            seed=params['seed']
        )

        Q_target.eval()

        utils_for_q_learning.sync_networks(
            target=Q_target,
            online=Q_object,
            alpha=params['target_network_learning_rate'],
            copy=True)

        if Q_object.is_env_discrete:
            params['policy_type'] = 'e_greedy'

        Q_objects.append(Q_object)
        Q_targets.append(Q_target)

    return Q_objects, Q_targets


def agent_networks_centralized(params, train_env, device):
    """
    Define online and offline networks
    :param params:
    :param train_env:
    :param device:
    :return:
    """

    Q_Constructor = None

    assert params['functional'] in ["fourier", "polynomial", "legendre"], "Functional type is not acceptable!"
    if params['functional'] == "fourier":
        Q_Constructor = FourierAgent
    elif params['functional'] == "polynomial":
        Q_Constructor = PolynomialAgent
    elif params['functional'] == "legendre":
        Q_Constructor = LegendreAgent

    Q_object = Q_Constructor(
        params,
        train_env,
        device=device,
        seed=params['seed'],
        centralized=True
    )
    Q_target = Q_Constructor(
        params,
        train_env,
        device=device,
        seed=params['seed'],
        centralized=True
    )

    Q_target.eval()

    utils_for_q_learning.sync_networks(
        target=Q_target,
        online=Q_object,
        alpha=params['target_network_learning_rate'],
        copy=True)

    if Q_object.is_env_discrete:
        params['policy_type'] = 'e_greedy'

    return Q_object, Q_target


def evaluate_model(env, Q_objects, t=100, log=1000):
    evaluation_rewards = []
    kk = False
    fail_count = 0
    for i in range(t):
        evaluation_reward = np.zeros(len(Q_objects))
        s_eval, done_eval = multiwalker_reset_separate(env), False
        steps_eval = 0
        max_r = 0
        min_r = 10
        while not done_eval and env.agents:
            actions_eval = {}
            for agent_index, agent_name in enumerate(env.agents):
                actions_eval[agent_name] = list(Q_objects[agent_index].e_greedy_policy(s_eval[agent_name],
                                                                                       1, steps_eval,
                                                                                       'test'))

            # environment step
            try:
                data_eval = multiwalker_step_separate(env, actions_eval)
            except Exception as e:
                print(e)
            s_eval_, r_eval, done_eval = data_eval[0], data_eval[1], data_eval[2]

            try:
                evaluation_reward += np.array(list(r_eval.values())[:len(Q_objects)])
            except Exception as e:
                print(e)
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

    evaluation_rewards = np.array(evaluation_rewards)
    eval_team = np.zeros(evaluation_rewards.shape[0])
    for i in range(0, np.array(evaluation_rewards).shape[1], 2):
        eval_team += evaluation_rewards[:, i]

    return eval_team.mean(), 1 - (fail_count/t)


def evaluate_model_centralized(env, Q_object, t=100, log=1000):
    evaluation_rewards = []
    kk = False
    fail_count = 0
    for i in range(t):
        evaluation_reward = np.zeros(env.n)
        s_eval, done_eval = multiwalker_reset_centralized(env), False
        max_r = 0
        min_r = 10
        steps_eval = 0
        while not done_eval and env.org_env.agents:

            act_eval = Q_object.e_greedy_policy(s_eval, 1, steps_eval, 'test')

            actions_eval = {}
            for agent_name, action in zip(env.agents, np.array_split(act_eval, env.n)):
                actions_eval[agent_name] = list(action)

            try:
                data = multiwalker_step_centralized(env, actions_eval)
            except Exception as e:
                print(e)

            s_eval_, r_eval, done_eval = data[0], data[1], data[2]

            evaluation_reward += r_eval
            s_eval = s_eval_
            if r_eval > max_r:
                max_r = r_eval

            if r_eval < min_r:
                min_r = r_eval

            if kk:
                env.render()

            steps_eval += 1

        if min_r < -50:
            fail_count += 1
        if i % log == 0:
            print(f'{i} Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, Minimum Reward: {min_r}')
        evaluation_rewards.append(evaluation_reward)

    print(f'fail_count: {fail_count}')


    return np.array(evaluation_rewards).mean(), 1 - (fail_count/t)


if __name__ == "__main__":
    _params = argument_parser()

    device = 'cuda'

    _params = argument_parser()

    _params['obs_indexes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30]

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
                                           forward_reward=5.0,
                                           # render_mode='human'
                                           )

    if _params['ma_type'] == 'centralized':

        _env = multiwalker_adjustment_centralized(_env, obs_indexes=_params['obs_indexes'])
        _Q_object, _ = agent_networks_centralized(_params, _env, device)
        _Q_object.load_state_dict(torch.load(_params['model_filepath'].replace('object_x', f'object')))
        reward, success = evaluate_model_centralized(_env, _Q_object, _params['nb_test'])

    else:

        _env = multiwalker_adjustment_separate(_env, obs_indexes=_params['obs_indexes'])

        _params["nb_agents"] = _env.n

        _Q_objects, _ = agent_networks(_params, _env, 'cpu')

        for i, _ in enumerate(_Q_objects):
            _Q_objects[i].load_state_dict(torch.load(_params['model_filepath'].replace('object_x', f'object_{i}')))
            # _Q_objects_mix[i].load_state_dict(torch.load(_params['model_filepath_mix'].replace('object_x', f'object_{i}')))

        reward, success = evaluate_model(_env, _Q_objects, _params['nb_test'])

    print(f'ma_type: {_params["ma_type"]}, '
          f'seed: {_params["seed"]}, '
          f'evaluation reward: {reward}, '
          f'success: {success}')


