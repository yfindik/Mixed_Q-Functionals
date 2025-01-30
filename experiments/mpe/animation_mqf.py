import sys
import os
from os.path import abspath, dirname

root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)
import argparse
import time
import numpy as np
import torch
from functional_critic import utils, utils_for_q_learning
from functional_critic.agents import FourierAgent, LegendreAgent, PolynomialAgent

from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios


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
    params['model_filepath'] = f'{args.path_to_model}/logs/seed_{args.seed}_object_x_steps_{args.step}'
    params['render'] = args.render
    params['nb_test'] = args.nb_test
    params['seed'] = args.seed
    params['step'] = args.step
    params['scenario'] = args.scenario

    return params


def agent_networks(params, env, device):
    """
    Define online and offline networks
    :param params:
    :param env:
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
        env,
        device=device,
        seed=params['seed']
    )
    Q_target = Q_Constructor(
        params,
        env,
        device=device,
        seed=params['seed']
    )

    return Q_object, Q_target


def agent_networks_mpe(params, env, device):
    """
    Define online and offline networks
    :param params:
    :param env:
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
            env,
            device=device,
            seed=params['seed']
        )
        Q_target = Q_Constructor(
            params,
            env,
            device=device,
            seed=params['seed']
        )

        Q_objects.append(Q_object)
        Q_targets.append(Q_target)

    return Q_objects, Q_targets


def evaluate_model_mpe(env, Q_objects, max_episode_len, seed, kk=True, t=10, log=1000):
    evaluation_rewards = []

    agents_done = []
    agents_collision_count = []
    for i in range(t):
        evaluation_reward = np.zeros(len(Q_objects))
        s_eval, done_eval = env.reset(), False
        steps_eval = 0
        while not done_eval:
            actions_eval = []
            for agent_index, Q_object in enumerate(Q_objects):
                actions_eval.append(
                    Q_object.e_greedy_policy(s_eval[agent_index], 1, steps_eval, 'test'))

            s_eval_, r_eval, done_eval, _ = env.step(actions_eval)
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

    # print(evaluation_rewards, np.array(agents_done)[:, 2].sum(), t)
    print(f'Seed: {seed} - Success Rate: {np.array(agents_done)[:, -1].mean() : .3f} ')
    print(f'Seed: {seed} - Collision Avg: {np.array(agents_collision_count)[:, -1].mean() : .3f}')
    print(f'Seed: {seed} - Team Reward Avg: {np.sum(np.array(evaluation_rewards), axis=1).mean(): .3f}')

    return np.array(agents_done)[:, -1].mean(), \
           np.array(agents_collision_count)[:, -1].mean(), \
           np.sum(np.array(evaluation_rewards), axis=1).mean()


def evaluate_model_mpe_catcher(env, Q_objects, max_episode_len, seed, kk=True, t=10, log=1000):
    evaluation_rewards = []
    agents_total_catch = []
    agents_catch_with_three = []
    agents_catch_with_two = []
    agents_catch_with_one = []
    for i in range(t):
        evaluation_reward = np.zeros(len(Q_objects))
        s_eval, done_eval = env.reset(), False
        steps_eval = 0
        while not done_eval:
            actions_eval = []
            for agent_index, Q_object in enumerate(Q_objects):
                actions_eval.append(
                    Q_object.e_greedy_policy(s_eval[agent_index], 1, steps_eval, 'test'))

            s_eval_, r_eval, done_eval, _ = env.step(actions_eval)
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
        # agents_total_catch.append(len(env.world.agents[-1].catchers) - 1)
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


if __name__ == "__main__":
    _params = argument_parser()

    scenario = scenarios.load(_params["env_name"] + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    _env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                         scenario.observation, width=700, height=700,
                         discrete_action_space=False
                         )

    _params["nb_agents"] = len(_env.action_space)

    _Q_objects, _ = agent_networks_mpe(_params, _env, 'cpu')
    for i, _ in enumerate(_Q_objects):
        _Q_objects[i].load_state_dict(
            torch.load(_params['model_filepath'].replace('object_x', f'object_{i}')))

    env_catcher = 'predator' in _params['scenario']

    if not env_catcher:

        success, collision, reward = evaluate_model_mpe(_env, _Q_objects, 50, _params['seed'],
                                                        kk=_params['render'], t=_params['nb_test'])

    else:

        evaluate_model_mpe_catcher(_env, _Q_objects, 50, _params['seed'],
                                   kk=_params['render'], t=_params['nb_test'])
