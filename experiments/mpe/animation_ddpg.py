import sys
import os
from os.path import abspath, dirname

root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)
import argparse
import numpy as np
import torch
from ddpg.agent import Agent
from functional_critic import utils, utils_for_q_learning
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
    params['model_filepath'] = f'{args.path_to_model}/logs'
    params['render'] = args.render
    params['nb_test'] = args.nb_test
    params['seed'] = args.seed
    params['steps'] = args.step
    params['scenario'] = args.scenario

    return params


def make_env(params, discrete_action_space=True):
    # pip install -e /home/yasin/Desktop/studies/multiagent-particle-envs

    # load scenario from script
    scenario = scenarios.load(params["env_name"] + ".py").Scenario()
    # create world
    world = scenario.make_world(seed=1)
    # create multiagent environment

    # switch w 1900, h = 1000

    # width = 900
    # height = 900

    width = 700
    height = 700

    train_env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                              width=width, height=height, discrete_action_space=discrete_action_space,
                              # done_callback=scenario.done,
                              )
    eval_env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                             width=width, height=height, discrete_action_space=discrete_action_space,
                             # done_callback=scenario.done,
                             )

    params['env'] = train_env
    utils_for_q_learning.set_random_seed(params)

    command_string = '"python ' + " ".join(sys.argv) + '"'
    params["command_string"] = command_string
    params["nb_agents"] = len(train_env.action_space)

    utils.save_hyper_parameters(params, params['seed'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return train_env, eval_env, device, params


def agent_networks_ddpg_2(train_env, params, device):
    obs_shape_n = train_env.observation_space[0].shape[0]
    act_shape_n = train_env.action_space[0].shape[0]
    max_action = train_env.action_space[0].high[0]
    agents = []
    for i in range(train_env.n):
        agents.append(Agent([obs_shape_n], act_shape_n, params, device))

    return agents


def agent_networks_ddpg_2_centralized(train_env, params, device):
    # this ddpg is working the other has some problem
    obs_shape_n = train_env.observation_space[0].shape[0] * train_env.n
    act_shape_n = train_env.action_space[0].shape[0] * train_env.n
    max_action = train_env.action_space[0].high[0]
    agent = Agent([obs_shape_n], act_shape_n, params, device)

    return agent


def evaluate_iddpg(agents, params, eval_env, total_steps, seed, kk=True, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    for agent_index, agent in enumerate(agents):
        _dir = os.path.join(path, f"seed_{seed}_object_{agent_index}_steps_" + str(total_steps))
        agent.load(_dir)
    evaluation_rewards = []
    agents_done = []
    agents_collision_count = []
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = eval_env.reset(), False
        steps_eval = 0
        while not done_eval:
            actions_eval = []
            for agent_index, agent in enumerate(agents):
                actions_eval.append(agent.select_action(np.array(s_eval[agent_index])))

            s_eval_, r_eval, done_eval, _ = eval_env.step(actions_eval)
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                eval_env.render()

            steps_eval += 1
            done_eval = False
            if params['max_episode_len'] and steps_eval == params['max_episode_len']:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)
        d = []
        for ii in eval_env.agents:
            d.append(ii.is_done)
        d.append(all(d))
        agents_done.append(d)

        collision_count = []
        for ii in eval_env.agents:
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


def evaluate_iddpg_catcher(agents, params, eval_env, total_steps, seed, kk=True, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    for agent_index, agent in enumerate(agents):
        _dir = os.path.join(path, f"seed_{seed}_object_{agent_index}_steps_" + str(total_steps))
        agent.load(_dir)
    evaluation_rewards = []
    agents_total_catch = []
    agents_catch_with_three = []
    agents_catch_with_two = []
    agents_catch_with_one = []
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = eval_env.reset(), False
        steps_eval = 0
        while not done_eval:
            actions_eval = []
            for agent_index, agent in enumerate(agents):
                actions_eval.append(agent.select_action(np.array(s_eval[agent_index])))

            s_eval_, r_eval, done_eval, _ = eval_env.step(actions_eval)
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                eval_env.render()

            steps_eval += 1
            done_eval = False
            if params['max_episode_len'] and steps_eval == params['max_episode_len']:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)
        threes = 0
        twos = 0
        ones = 0
        for ii in eval_env.world.agents[-1].catchers:
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


def evaluate_cddpg(agent, params, eval_env, total_steps, seed, kk=True, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    # for agent_index, agent in enumerate(agents):
    _dir = os.path.join(path, f"seed_{seed}_object_{0}_steps_" + str(total_steps))
    agent.load(_dir)
    evaluation_rewards = []
    agents_done = []
    agents_collision_count = []
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = eval_env.reset(), False
        s_eval = np.array(s_eval).flatten()
        steps_eval = 0
        while not done_eval:
            actions_eval = agent.select_action(s_eval)

            s_eval_, r_eval, done_eval, _ = eval_env.step(actions_eval.reshape((eval_env.n, 2)))
            s_eval_ = np.array(s_eval_).flatten()
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                eval_env.render()

            steps_eval += 1
            done_eval = False
            if params['max_episode_len'] and steps_eval == params['max_episode_len']:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)
        d = []
        for ii in eval_env.agents:
            d.append(ii.is_done)
        d.append(all(d))
        agents_done.append(d)

        collision_count = []
        for ii in eval_env.agents:
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


def evaluate_cddpg_catcher(agent, params, eval_env, total_steps, seed, kk=True, t=10, log=1000):
    path = os.path.join(params["experiment_filepath"], "logs")
    # for agent_index, agent in enumerate(agents):
    _dir = os.path.join(path, f"seed_{seed}_object_{0}_steps_" + str(total_steps))
    agent.load(_dir)
    evaluation_rewards = []
    agents_total_catch = []
    agents_catch_with_three = []
    agents_catch_with_two = []
    agents_catch_with_one = []
    for i in range(t):
        evaluation_reward = np.zeros(eval_env.n)
        s_eval, done_eval = eval_env.reset(), False
        s_eval = np.array(s_eval).flatten()
        steps_eval = 0
        while not done_eval:
            actions_eval = agent.select_action(s_eval)

            s_eval_, r_eval, done_eval, _ = eval_env.step(actions_eval.reshape((eval_env.n, 2)))
            s_eval_ = np.array(s_eval_).flatten()
            evaluation_reward += r_eval
            s_eval = s_eval_

            if kk:
                eval_env.render()

            steps_eval += 1
            done_eval = False
            if params['max_episode_len'] and steps_eval == params['max_episode_len']:
                done_eval = True
        evaluation_rewards.append(evaluation_reward)
        threes = 0
        twos = 0
        ones = 0
        for ii in eval_env.world.agents[-1].catchers:
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
    # _device = 'cpu'

    env_catcher = 'predator' in _params['scenario']
    if not env_catcher:
        _train_env, _eval_env, _device, _params = make_env(_params, discrete_action_space=False)

        if _params['ma_type'] == 'iddpg':
            _agents = agent_networks_ddpg_2(_train_env, _params, _device)
            success, collision, reward = evaluate_iddpg(_agents, _params, _eval_env,
                                                        total_steps=_params['steps'], seed=_params['seed'],
                                                        kk=_params['render'], t=_params['nb_test'])

        elif _params['ma_type'] == 'cddpg':
            _agent = agent_networks_ddpg_2_centralized(_train_env, _params, _device)
            success, collision, reward = evaluate_cddpg(_agent, _params, _eval_env,
                                                        total_steps=_params['steps'], seed=_params['seed'],
                                                        kk=_params['render'], t=_params['nb_test'])

        print(f'ma_type: {_params["ma_type"]}, '
              f'seed: {_params["seed"]}, '
              f'evaluation reward: {reward}, '
              f'success: {success}')


    else:
        _train_env, _eval_env, _device, _params = make_env(_params, discrete_action_space=False)

        if _params['ma_type'] == 'iddpg':
            _agents = agent_networks_ddpg_2(_train_env, _params, _device)
            total_catch_per_episode, \
            three_agents_catch_per_episode, \
            two_agents_catch_per_episode, \
            one_agents_catch_per_episode, \
            reward = \
                evaluate_iddpg_catcher(_agents, _params, _eval_env,
                                       total_steps=_params['steps'], seed=_params['seed'],
                                       kk=_params['render'], t=_params['nb_test'])

        elif _params['ma_type'] == 'cddpg':
            _agents = agent_networks_ddpg_2_centralized(_train_env, _params, _device)
            total_catch_per_episode, \
            three_agents_catch_per_episode, \
            two_agents_catch_per_episode, \
            one_agents_catch_per_episode, \
            reward = \
                evaluate_cddpg_catcher(_agents, _params, _eval_env,
                                       total_steps=_params['steps'], seed=_params['seed'],
                                       kk=_params['render'], t=_params['nb_test'])

        print(
            f'Catch per Eps: {total_catch_per_episode:.3f} +- '
            f'Three Agents Catch per Eps: {three_agents_catch_per_episode:.3f} +- '
            f'Two Agents Catch per Eps: {two_agents_catch_per_episode:.3f} +- '
            f'One Agents Catch per Eps: {one_agents_catch_per_episode:.3f} +- '
            f'Rewards: {reward:.3f}')
