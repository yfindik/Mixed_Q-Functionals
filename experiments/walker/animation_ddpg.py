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

from ddpg.agent import Agent
from functional_critic import utils, utils_for_q_learning
from logging_utils import MetaLogger
import datetime

from walker_utils_ma import multiwalker_reset_separate, multiwalker_step_separate, multiwalker_adjustment_centralized, \
    multiwalker_adjustment_separate, multiwalker_reset_centralized, multiwalker_step_centralized


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
    params['model_filepath'] = f'{args.path_to_model}/logs'
    params['render'] = args.render
    params['nb_test'] = args.nb_test
    params['seed'] = args.seed
    params['steps'] = args.step

    return params


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
                                           forward_reward=params['forward_reward'],
                                           render_mode='human')
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


def agent_networks_ddpg_2(train_env, params, device):
    # this ddpg is working the other has some problem
    obs_shape_n = list(train_env.observation_spaces.values())[0].shape[0]
    act_shape_n = list(train_env.action_spaces.values())[0].shape[0]
    max_action = list(train_env.action_spaces.values())[0].high[0]
    agents = []
    for i in range(train_env.n):
        agents.append(Agent([obs_shape_n], act_shape_n, params, device, max_action))

    return agents


def agent_networks_ddpg_2_centralized(train_env, params, device):
    # this ddpg is working the other has some problem
    obs_shape_n = list(train_env.observation_spaces.values())[0].shape[0] * train_env.n
    act_shape_n = list(train_env.action_spaces.values())[0].shape[0] * train_env.n
    max_action = list(train_env.action_spaces.values())[0].high[0]
    agent = Agent([obs_shape_n], act_shape_n, params, device)

    return agent


def evaluate_iddpg(agents, params, eval_env, total_steps, seed, kk=False, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    for agent_index, agent in enumerate(agents):
        _dir = os.path.join(path, f"seed_{seed}_object_{agent_index}_steps_" + str(total_steps))
        agent.load(_dir)
    evaluation_rewards = []

    fail_count = 0
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = multiwalker_reset_separate(eval_env), False
        steps_eval = 0
        max_r = 0
        min_r = 10
        while not done_eval and eval_env.org_env.agents:
            actions_eval = {}

            for agent_index, agent_name in enumerate(eval_env.agents):
                actions_eval[agent_name] = list(agents[agent_index].select_action(np.array(s_eval[agent_name])))

            try:
                data = multiwalker_step_separate(eval_env, actions_eval)
            except Exception as e:
                print(e)
            s_eval_, r_eval, done_eval, = data[0], data[1], data[2]

            try:
                evaluation_reward += np.array(list(r_eval.values())[:eval_env.n])
            except Exception as e:
                print(e)
            s_eval = s_eval_

            if list(r_eval.values())[0] > max_r:
                max_r = list(r_eval.values())[0]

            if list(r_eval.values())[0] < min_r:
                min_r = list(r_eval.values())[0]

            if kk:
                eval_env.render()

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

    return eval_team.mean(), 1 - (fail_count / t)


def evaluate_cddpg(agent, params, eval_env, total_steps, seed, kk=False, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    # for agent_index, agent in enumerate(agents):
    _dir = os.path.join(path, f"seed_{seed}_object_{0}_steps_" + str(total_steps))
    agent.load(_dir)
    evaluation_rewards = []
    fail_count = 0
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = multiwalker_reset_centralized(eval_env), False
        s_eval = np.array(s_eval).flatten()
        max_r = 0
        min_r = 10
        steps_eval = 0
        while not done_eval:
            act_eval = agent.select_action(s_eval, noise=False)

            actions_eval = {}
            for agent_name, action in zip(eval_env.agents, np.array_split(act_eval, eval_env.n)):
                actions_eval[agent_name] = list(action)

            try:
                data = multiwalker_step_centralized(eval_env, actions_eval)
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
                eval_env.render()

            steps_eval += 1

        if min_r < -50:
            fail_count += 1
        if i % log == 0:
            print(f'{i} Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, Minimum Reward: {min_r}')
        evaluation_rewards.append(evaluation_reward)

    print(f'fail_count: {fail_count}')

    return np.array(evaluation_rewards).mean(), 1 - (fail_count / t)


if __name__ == "__main__":

    _params = argument_parser()
    _params['obs_indexes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30]

    _train_env, _eval_env, _device, _params = make_walker(_params, adj=True,
                                                          centralized=_params['ma_type'] == 'cddpg')

    if _params['ma_type'] == 'iddpg':
        _agents = agent_networks_ddpg_2(_train_env, _params, _device)
        reward, success = evaluate_iddpg(_agents, _params, _eval_env if _params['render'] else _train_env,
                                         total_steps=_params['steps'], seed=_params['seed'], kk=False, t=10)

    elif _params['ma_type'] == 'cddpg':
        _agent = agent_networks_ddpg_2_centralized(_train_env, _params, _device)
        reward, success = evaluate_cddpg(_agent, _params, _eval_env if _params['render'] else _train_env,
                                         total_steps=_params['steps'], seed=_params['seed'], kk=False, t=10000)


    print(f'ma_type: {_params["ma_type"]}, '
          f'seed: {_params["seed"]}, '
          f'evaluation reward: {reward}, '
          f'success: {success}')

