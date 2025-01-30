import sys
import os
from os.path import abspath, dirname
root_dir = dirname(dirname(dirname(abspath(__file__))))
os.chdir(root_dir)
sys.path.append(root_dir)
import argparse
from time import time
import numpy as np
import torch
from ddpg.agent import Agent
from functional_critic import utils, utils_for_q_learning
from logging_utils import MetaLogger
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
import datetime
import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def argument_parser():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper_param_directory",
                        required=False,
                        default="hyper_parameters",
                        type=str)

    parser.add_argument("--hyper_parameters_name",
                        required=False,
                        help="0, 10, 20, etc. Corresponds to .hyper file",
                        default="global",
                        type=str)  # OpenAI gym environment name

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
                        required=True)

    parser.add_argument("--ma_type",
                        type=str,
                        help="Experiment Type",
                        required=True)

    parser.add_argument("--run_title",
                        type=str,
                        help="subdirectory for this run",
                        required=False,
                        default="sum_run")

    parser.add_argument("--team_reward",
                        action='store_true')

    parser.add_argument("--daemon",
                        action='store_true')

    parser.add_argument("--seed", default=0,
                        type=int)

    parser.add_argument("--update_step", default=50,
                        type=int)

    parser.add_argument("--nb_runs", default=5,
                        type=int)

    parser.add_argument("--max_episode", default=2000,
                        type=int)

    parser.add_argument("--max_step", default=10000,
                        type=int)

    parser.add_argument("--evaluation_frequency", default=10000,
                        required=False, type=int)

    parser.add_argument("--saving_frequency", default=10000,
                        required=False, type=int)

    parser.add_argument("--max_episode_len", default=None,
                        required=False, type=int)

    parser.add_argument("--is_not_gym_env", action='store_true')

    parser.add_argument("--save_model", action="store_true")

    args, unknown = parser.parse_known_args()
    other_args = {(utils.remove_prefix(key, '--'), val)
                  for (key, val) in zip(unknown[::2], unknown[1::2])}

    params = utils.get_hyper_parameters(args.hyper_parameters_name,
                                        args.hyper_param_directory)

    full_experiment_name = os.path.join(args.experiment_name, args.run_title)
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))
    params["hyperparams_dir"] = hyperparams_dir
    params["hyper_parameters_name"] = args.hyper_parameters_name
    params['seed'] = args.seed
    params['update_step'] = args.update_step
    params['team_reward'] = args.team_reward
    params['nb_runs'] = args.nb_runs
    params['daemon'] = args.daemon
    params['ma_type'] = args.ma_type
    params['max_episode'] = args.max_episode
    params['max_step'] = args.max_step
    params['is_not_gym_env'] = args.is_not_gym_env
    params["full_experiment_name"] = full_experiment_name
    params['evaluation_frequency'] = args.evaluation_frequency
    params['saving_frequency'] = args.saving_frequency
    params['max_episode_len'] = args.max_episode_len
    params['save_model'] = args.save_model or True
    params['experiment_filepath'] = os.path.join(os.getcwd(), os.path.join(args.experiment_name, args.run_title))
    params['experiment_name'] = args.experiment_name
    for arg_name, arg_value in other_args:
        utils.update_param(params, arg_name, arg_value)

    return params


def meta_logger_initialization(full_experiment_name, seed):
    """
    The logging utils
    :return:
    """
    meta_logger = MetaLogger(full_experiment_name)
    logging_filename = f"seed_{seed}.pkl"
    meta_logger.add_field("average_loss", logging_filename)
    meta_logger.add_field("average_q", logging_filename)
    meta_logger.add_field("average_q_star", logging_filename)
    meta_logger.add_field("episodic_rewards", logging_filename)
    meta_logger.add_field("episodic_rewards_avg_over_log", logging_filename)
    meta_logger.add_field("evaluation_rewards", logging_filename)
    meta_logger.add_field("all_times", logging_filename)
    meta_logger.add_field("episodes_so_far", logging_filename)
    return meta_logger


def print_out_configurations(params):
    """

    Print out the configurations.
    :param params:
    :return:
    """

    if params['daemon']:
        log_to_file(params['daemon_fname'], "{:=^100s}".format("Basic Configurations"))
        log_to_file(params['daemon_fname'], f"Daemon Filename: {params['daemon_fname']}")
        log_to_file(params['daemon_fname'], f"Training Environment: {params['env_name']}")
        log_to_file(params['daemon_fname'], f"Functional: {params['functional']}")
        log_to_file(params['daemon_fname'], f"rank: {params['rank']}")
        log_to_file(params['daemon_fname'], f"seed: {params['seed']}")
        log_to_file(params['daemon_fname'], f"learning_rate: {params['learning_rate']}")
        log_to_file(params['daemon_fname'], f"target_network_learning_rate: {params['target_network_learning_rate']}")

        log_to_file(params['daemon_fname'], f"nb_runs: {params['nb_runs']}")
        log_to_file(params['daemon_fname'], f"ma_type: {params['ma_type']}")
        log_to_file(params['daemon_fname'], f"policy_type: {params['policy_type']}")
        log_to_file(params['daemon_fname'], f"noise_std: {params['noise_std']}")

        log_to_file(params['daemon_fname'], "{:=^100s}".format("Model Configurations"))
        log_to_file(params['daemon_fname'], f"TD3 Trick:: {params['minq']}")
        log_to_file(params['daemon_fname'], f"Entropy Regularization: {params['entropy_regularized']}")
        log_to_file(params['daemon_fname'], "{:=^100s}".format("Sampling Configurations"))
        log_to_file(params['daemon_fname'], f": {params['env_name']}")

        log_to_file(params['daemon_fname'],
                    f"Using quantile sampling during bootstrapping:: {params['use_quantile_sampling_bootstrapping']}")
        log_to_file(params['daemon_fname'],
                    f"Using quantile sampling during evaluation interaction: {params['use_quantile_sampling_evaluation_interaction']}")
        log_to_file(params['daemon_fname'],
                    f"Using quantile sampling during training interaction:: {params['use_quantile_sampling_training_interaction']}")
        log_to_file(params['daemon_fname'], f"Sampling Percent: {params['quantile_sampling_percent']}")
        log_to_file(params['daemon_fname'], f"Anneal Sampling Percent: {params['anneal_quantile_sampling']}")
        log_to_file(params['daemon_fname'], "{:=^100s}".format("Split"))

    else:
        print("{:=^100s}".format("Basic Configurations"))
        print("Training Environment:", params['env_name'])
        print("Functional:", params['functional'])
        print("rank:", params['rank'])
        print("seed:", params['seed'])
        print("learning_rate:", params['learning_rate'])
        print("nb_runs:", params['nb_runs'])
        print("ma_type:", params['ma_type'])
        print("policy_type:", _params["policy_type"])

        print("{:=^100s}".format("Model Configurations"))
        print("TD3 Trick:", params['minq'])
        print("Entropy Regularization:", params['entropy_regularized'])

        print("{:=^100s}".format("Sampling Configurations"))
        print("Using quantile sampling during bootstrapping:", params['use_quantile_sampling_bootstrapping'])
        print("Using quantile sampling during evaluation interaction:",
              params['use_quantile_sampling_evaluation_interaction'])
        print("Using quantile sampling during training interaction:",
              params['use_quantile_sampling_training_interaction'])
        print("Sampling Percent:", params['quantile_sampling_percent'])
        print("Anneal Sampling Percent:", params['anneal_quantile_sampling'])

        print("{:=^100s}".format("Split"))


def make_env(params, discrete_action_space=True):
    # load scenario from script
    scenario = scenarios.load(params["env_name"] + ".py").Scenario()
    # create world
    world = scenario.make_world()
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
        if params['daemon']:
            log_to_file(params['daemon_fname'], "running on CUDA...")
        else:
            print("running on CUDA...")
    else:
        device = torch.device("cpu")
        if params['daemon']:
            log_to_file(params['daemon_fname'], "running on CPU...")
        else:
            print("running on CPU...")

    return train_env, eval_env, device, params


def agent_buffer_initialization_mpe(train_env, Q_objects, warm_up_steps, max_episode_len):
    print("Start the Initialization Process!")
    steps = 0
    while steps < warm_up_steps:
        s, done, t = train_env.reset(), False, 0
        while not done:
            actions = []
            # for action_space in train_env.action_space:
            # actions.append(action_space.sample())
            for i, Q_object in enumerate(Q_objects):
                actions.append(Q_object.enact_policy(s[i], 1, 1, 'test'))

            s_, r, done, info = train_env.step(actions)

            for agent_index, Q_object in enumerate(Q_objects):
                Q_object.buffer_object.append(s[agent_index], actions[agent_index],
                                              r[agent_index], done[agent_index], s_[agent_index])

            s = s_
            steps += 1
            done = False
            if max_episode_len and steps % max_episode_len == 0:
                done = True

    print("The initialization process finished!")


def log_to_file(fname, text):
    with open(fname, "a") as fh:
        fh.write(str(datetime.datetime.now()) + ' ' + text + "\n")


def set_run_title(params):
    # params['max_episode']
    if params['ma_type'] == 'maddpg':
        t = 'team-reward_' if params['team_reward'] else 'individual-reward_'
        run_title = f"{params['ma_type']}_run_" \
                    f"{params['update_step']}-step-update_" \
                    f"{t}" \
                    f"episode-{params['max_episode']}_" \
                    f"lr{str(params['learning_rate']).replace('.', '')}_b{params['batch_size']}_l{params['num_layers']}_" \
                    f"n{params['layer_size']}_m{params['max_buffer_size']}_iter1"
    else:
        run_title = f"{params['ma_type']}_run_" \
                    f"{params['update_step']}-step-update_" \
                    f"func-{params['functional']}_rank-{params['rank']}_episode-{params['max_episode']}_" \
                    f"lr{str(params['learning_rate']).replace('.', '')}_b{params['batch_size']}_l{params['num_layers']}_" \
                    f"n{params['layer_size']}_m{params['max_buffer_size']}_iter1"

    full_experiment_name = os.path.join(params['experiment_name'], run_title)
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    params["full_experiment_name"] = full_experiment_name
    params["hyperparams_dir"] = hyperparams_dir
    params['experiment_filepath'] = os.path.join(os.getcwd(), os.path.join(params['experiment_name'], run_title))


# def agent_networks_ddpg(train_env, params, device):
#     obs_shape_n = train_env.observation_space[0].shape[0]
#     act_shape_n = train_env.action_space[0].shape[0]
#     max_action = train_env.action_space[0].high[0]
#     agents = []
#     for i in range(train_env.n):
#         agents.append(DDPG(obs_shape_n, act_shape_n, max_action, device, params))
#
#     return agents
#

def agent_networks_ddpg_2(train_env, params, device):
    # this ddpg is working the other has some problem
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


def train_maddpg(train_env, eval_env, meta_logger,
                 max_step, evaluation_frequency, saving_frequency, experiment_filepath,
                 seed, max_episode_len, learning_starts, save_model=True, max_checkpoints_to_keep=500,
                 daemon_fname='', update_step=50, team_reward=True):
    mem_large_enough = False
    with U.single_threaded_session():
        trainers = agent_networks_maddpg(_train_env, _params)

        # Initialize
        U.initialize()

        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

        total_steps = 0
        iter_start_time = time()
        episodes = 0
        episodic_rewards = []
        basis_statistics = []
        fp = 0
        sp = 0

        print('Starting iterations...')
        while total_steps < max_step:
            # get action
            obs_n, done = train_env.reset(), False
            episodic_reward = np.zeros(len(trainers))
            steps = 0
            while not done:
                fp_t = time()
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                # environment step
                new_obs_n, rew_n, done_n, info_n = train_env.step(action_n)

                # collect experience
                terminal = (steps >= max_episode_len)
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

                # filling the memory first
                if not mem_large_enough and total_steps < learning_starts:
                    total_steps += 1
                    continue
                elif not mem_large_enough and total_steps == learning_starts:
                    total_steps = 0
                    mem_large_enough = True

                fp += time() - fp_t

                obs_n = new_obs_n
                sp_t = time()
                # update all trainers, if not in display or benchmark mode
                basis_statistics_step = []
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    basis_statistics_step.append(agent.update(trainers, total_steps,
                                                              team_reward=team_reward, update_step=update_step))
                    # [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

                basis_statistics.append(basis_statistics_step)
                sp += time() - sp_t
                if (total_steps % evaluation_frequency == 0) or (total_steps == max_step - 1):
                    evaluation_rewards = []
                    kk = False
                    for _ in range(10):
                        evaluation_reward = np.zeros(len(trainers))
                        s_eval, done_eval = eval_env.reset(), False
                        steps_eval = 0
                        while not done_eval:
                            action_eval_n = [agent.action(obs) for agent, obs in zip(trainers, s_eval)]

                            s_eval_, r_eval, done_eval, _ = eval_env.step(action_eval_n)
                            evaluation_reward += r_eval
                            s_eval = s_eval_

                            if kk:
                                eval_env.render()

                            steps_eval += 1
                            done_eval = False
                            if max_episode_len and steps_eval == max_episode_len:
                                done_eval = True
                        evaluation_rewards.append(evaluation_reward)

                    if daemon_fname != '':
                        log_to_file(daemon_fname, f"Episodes {episodes}:, ")
                        log_to_file(daemon_fname, f"Total steps {total_steps}/{max_step}, ")
                        log_to_file(daemon_fname, f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)},")
                        log_to_file(daemon_fname, f"Time: {round(time() - iter_start_time, 3)}")
                    else:
                        print(f"Episodes {episodes}:, "
                              f"Total steps {total_steps}/{max_step}, "
                              f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)}, "  # mean per agent
                              f"Time: {round(time() - iter_start_time, 3)}, "
                              f"FP: {round(fp, 3)}, "
                              f"SP: {round(sp, 3)}")

                    """
                    Consolidate episode statistics
                    """
                    # mean_per_step_loss = np.nan_to_num(np.mean(np.array(per_step_losses), axis=0), nan=0)
                    # mean_qs = np.nan_to_num(np.mean(np.array(qs), axis=0), nan=0)
                    # mean_q_stars = np.nan_to_num(np.mean(np.array(q_stars), axis=0), nan=0)
                    iter_total_time = time() - iter_start_time
                    """
                    Update meta logger to record some statistics
                    """
                    meta_logger.append_datapoint("evaluation_rewards", np.mean(evaluation_rewards, axis=0), write=True)
                    if episodic_rewards:
                        mean_episodic_reward = np.nan_to_num(np.mean(np.array(episodic_rewards), axis=0), nan=0)
                        meta_logger.append_datapoint("episodic_rewards_avg_over_log", mean_episodic_reward, write=True)
                    meta_logger.append_datapoint("episodic_rewards", episodic_rewards, write=True)
                    meta_logger.append_datapoint("episodes_so_far", episodes, write=True)
                    # meta_logger.append_datapoint("average_loss", mean_per_step_loss, write=True)
                    # meta_logger.append_datapoint("average_q", mean_qs, write=True)
                    # meta_logger.append_datapoint("average_q_star", mean_q_stars, write=True)
                    meta_logger.append_datapoint("all_times", iter_total_time, write=True)

                    """
                    Reset tracking quantities
                    """
                    episodic_rewards, basis_statistics = [], []
                    iter_start_time = time()
                    fp = 0
                    sp = 0

                if save_model and ((total_steps % saving_frequency == 0) or total_steps == (max_step - 1)):
                    path = os.path.join(experiment_filepath, f"logs/seed_{seed}")
                    if not os.path.exists(path):
                        try:
                            os.makedirs(path, exist_ok=True)
                        except OSError:
                            print("Creation of the directory %s failed" % path)
                        else:
                            print("Successfully created the directory %s " % path)
                    U.save_state(f'{path}/steps_{total_steps + 1}', saver=saver)

                steps += 1
                total_steps += 1
                episodic_reward += rew_n

                # done = all(done_n)
                done = False
                if max_episode_len and steps == max_episode_len:
                    done = True

            episodes += 1
            # epsilons.append(eps)
            episodic_rewards.append(episodic_reward)
        U.reset()


def train_mpe_iddpg(agents, train_env, eval_env, meta_logger,
                    max_step, evaluation_frequency, saving_frequency,
                    experiment_filepath, seed, max_episode_len, learning_starts,
                    save_model=True, daemon_fname='', update_step=50):
    mem_large_enough = False
    total_steps = 0
    episodes = 0
    # epsilons = []
    per_step_losses = []
    qs = []
    q_stars = []
    episodic_rewards = []
    iter_start_time = time()
    fp = 0
    sp = 0
    while total_steps < max_step:
        s, done = train_env.reset(), False

        episodic_reward = np.zeros(train_env.n)
        steps = 0
        while not done and steps < max_episode_len:
            actions = []
            fp_t = time()
            for i, agent in enumerate(agents):
                actions.append(agent.select_action(np.array(s[i]), noise=True))

            s_, r, done, info = train_env.step(np.array(actions))

            for i, agent in enumerate(agents):
                # agent.replay_buffer.push((s[i], s_[i], actions[i], r[i], done[i]))
                agent.replay_buffer.push(s[i], actions[i], r[i], s_[i], int(done[i]))

            steps += 1
            total_steps += 1
            episodic_reward += r
            done = False
            if max_episode_len and steps == max_episode_len:
                done = True

            # filling the memory first
            if not mem_large_enough and total_steps < learning_starts:
                continue
            elif not mem_large_enough and total_steps == learning_starts:
                total_steps = 0
                episodes = 0
                episodic_rewards = []
                mem_large_enough = True

            fp += time() - fp_t

            s = s_

            losses = []
            basis_statistics = []
            agents_q = []
            agents_q_star = []

            sp_t = time()

            if total_steps % update_step == 0:  # only update every 100 steps

                for agent in agents:
                    agent.update()

            sp += time() - sp_t

            per_step_losses.append(losses)
            qs.append(agents_q)
            q_stars.append(agents_q_star)

            if (total_steps % evaluation_frequency == 0) or (total_steps == max_step - 1):
                evaluation_rewards = []
                kk = False
                for _ in range(10):
                    evaluation_reward = np.zeros(eval_env.n)
                    s_eval, done_eval = eval_env.reset(), False
                    steps_eval = 0
                    while not done_eval:
                        actions_eval = []
                        for i, agent in enumerate(agents):
                            actions_eval.append(agent.select_action(np.array(s_eval[i])))

                        s_eval_, r_eval, done_eval, _ = eval_env.step(np.array(actions_eval))
                        evaluation_reward += r_eval
                        s_eval = s_eval_

                        if kk:
                            eval_env.render()

                        steps_eval += 1
                        done_eval = False
                        if max_episode_len and steps_eval == max_episode_len:
                            done_eval = True
                    evaluation_rewards.append(evaluation_reward)

                if daemon_fname != '':
                    log_to_file(daemon_fname, f"Episodes {episodes}:, ")
                    log_to_file(daemon_fname, f"Total steps {total_steps}/{max_step}, ")
                    log_to_file(daemon_fname, f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)},")
                    log_to_file(daemon_fname, f"Time: {round(time() - iter_start_time, 3)}")
                else:
                    print(f"Episodes {episodes}:, "
                          f"Total steps {total_steps}/{max_step}, "
                          f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)}, "  # mean per agent
                          f"Time: {round(time() - iter_start_time, 3)}, "
                          f"FP: {round(fp, 3)}, "
                          f"SP: {round(sp, 3)}")

                """
                Consolidate episode statistics
                """
                mean_per_step_loss = np.nan_to_num(np.mean(np.array(per_step_losses), axis=0), nan=0)
                mean_qs = np.nan_to_num(np.mean(np.array(qs), axis=0), nan=0)
                mean_q_stars = np.nan_to_num(np.mean(np.array(q_stars), axis=0), nan=0)
                iter_total_time = time() - iter_start_time
                """
                Update meta logger to record some statistics
                """
                meta_logger.append_datapoint("evaluation_rewards", np.mean(evaluation_rewards, axis=0), write=True)
                if episodic_rewards:
                    mean_episodic_reward = np.nan_to_num(np.mean(np.array(episodic_rewards), axis=0), nan=0)
                    meta_logger.append_datapoint("episodic_rewards_avg_over_log", mean_episodic_reward, write=True)
                meta_logger.append_datapoint("episodic_rewards", episodic_rewards, write=True)
                meta_logger.append_datapoint("episodes_so_far", episodes, write=True)
                meta_logger.append_datapoint("average_loss", mean_per_step_loss, write=True)
                meta_logger.append_datapoint("average_q", mean_qs, write=True)
                meta_logger.append_datapoint("average_q_star", mean_q_stars, write=True)
                meta_logger.append_datapoint("all_times", iter_total_time, write=True)

                """
                Reset tracking quantities
                """
                episodic_rewards, per_step_losses, qs, q_stars = [], [], [], []
                iter_start_time = time()
                fp = 0
                sp = 0

            if save_model and ((total_steps % saving_frequency == 0) or total_steps == (max_step - 1)):
                path = os.path.join(experiment_filepath, "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                for agent_index, agent in enumerate(agents):
                    _dir = os.path.join(path, f"seed_{seed}_object_{agent_index}_steps_" + str(total_steps + 1))
                    agent.save(_dir)

        episodes += 1
        # epsilons.append(eps)
        episodic_rewards.append(episodic_reward)


def train_mpe_cddpg(agent, train_env, eval_env, meta_logger,
                    max_step, evaluation_frequency, saving_frequency,
                    experiment_filepath, seed, max_episode_len, learning_starts,
                    save_model=True, daemon_fname='', update_step=50):
    mem_large_enough = False
    total_steps = 0
    episodes = 0
    # epsilons = []
    per_step_losses = []
    qs = []
    q_stars = []
    episodic_rewards = []
    iter_start_time = time()
    fp = 0
    sp = 0
    while total_steps < max_step:
        s, done = train_env.reset(), False
        s = np.array(s).flatten()
        episodic_reward = np.zeros(train_env.n)
        steps = 0
        while not done and steps < max_episode_len:

            fp_t = time()

            actions = agent.select_action(s, noise=True)

            s_, r, done, info = train_env.step(actions.reshape((train_env.n, 2)))
            s_ = np.array(s_).flatten()

            agent.replay_buffer.push(s, actions, sum(r), s_, all(done))
            s = s_
            steps += 1
            total_steps += 1
            episodic_reward += r
            done = False
            if max_episode_len and steps == max_episode_len:
                done = True

            # filling the memory first
            if not mem_large_enough and total_steps < learning_starts:
                continue
            elif not mem_large_enough and total_steps == learning_starts:
                total_steps = 0
                episodes = 0
                episodic_rewards = []
                mem_large_enough = True

            fp += time() - fp_t

            losses = []
            basis_statistics = []
            agents_q = []
            agents_q_star = []

            sp_t = time()

            if total_steps % update_step == 0:  # only update every 100 steps

                agent.update()

            sp += time() - sp_t

            per_step_losses.append(losses)
            qs.append(agents_q)
            q_stars.append(agents_q_star)

            if (total_steps % evaluation_frequency == 0) or (total_steps == max_step - 1):
                evaluation_rewards = []
                kk = False
                for _ in range(10):
                    evaluation_reward = np.zeros(eval_env.n)
                    s_eval, done_eval = eval_env.reset(), False
                    s_eval = np.array(s_eval).flatten()

                    steps_eval = 0
                    while not done_eval:
                        actions_eval = agent.select_action(s_eval)

                        s_eval_, r_eval, done_eval, _ = eval_env.step(actions_eval.reshape((train_env.n, 2)))
                        s_eval_ = np.array(s_eval_).flatten()
                        evaluation_reward += r_eval
                        s_eval = s_eval_

                        if kk:
                            eval_env.render()

                        steps_eval += 1
                        done_eval = False
                        if max_episode_len and steps_eval == max_episode_len:
                            done_eval = True
                    evaluation_rewards.append(evaluation_reward)

                if daemon_fname != '':
                    log_to_file(daemon_fname, f"Episodes {episodes}:, ")
                    log_to_file(daemon_fname, f"Total steps {total_steps}/{max_step}, ")
                    log_to_file(daemon_fname, f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)},")
                    log_to_file(daemon_fname, f"Time: {round(time() - iter_start_time, 3)}")
                else:
                    print(f"Episodes {episodes}:, "
                          f"Total steps {total_steps}/{max_step}, "
                          f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)}, "  # mean per agent
                          f"Time: {round(time() - iter_start_time, 3)}, "
                          f"FP: {round(fp, 3)}, "
                          f"SP: {round(sp, 3)}")

                """
                Consolidate episode statistics
                """
                mean_per_step_loss = np.nan_to_num(np.mean(np.array(per_step_losses), axis=0), nan=0)
                mean_qs = np.nan_to_num(np.mean(np.array(qs), axis=0), nan=0)
                mean_q_stars = np.nan_to_num(np.mean(np.array(q_stars), axis=0), nan=0)
                iter_total_time = time() - iter_start_time
                """
                Update meta logger to record some statistics
                """
                meta_logger.append_datapoint("evaluation_rewards", np.mean(evaluation_rewards, axis=0), write=True)
                if episodic_rewards:
                    mean_episodic_reward = np.nan_to_num(np.mean(np.array(episodic_rewards), axis=0), nan=0)
                    meta_logger.append_datapoint("episodic_rewards_avg_over_log", mean_episodic_reward, write=True)
                meta_logger.append_datapoint("episodic_rewards", episodic_rewards, write=True)
                meta_logger.append_datapoint("episodes_so_far", episodes, write=True)
                meta_logger.append_datapoint("average_loss", mean_per_step_loss, write=True)
                meta_logger.append_datapoint("average_q", mean_qs, write=True)
                meta_logger.append_datapoint("average_q_star", mean_q_stars, write=True)
                meta_logger.append_datapoint("all_times", iter_total_time, write=True)

                """
                Reset tracking quantities
                """
                episodic_rewards, per_step_losses, qs, q_stars = [], [], [], []
                iter_start_time = time()
                fp = 0
                sp = 0

            if save_model and ((total_steps % saving_frequency == 0) or total_steps == (max_step - 1)):
                path = os.path.join(experiment_filepath, "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)

                _dir = os.path.join(path, f"seed_{seed}_object_{0}_steps_" + str(total_steps + 1))
                agent.save(_dir)

        episodes += 1
        # epsilons.append(eps)
        episodic_rewards.append(episodic_reward)


if __name__ == "__main__":

    _params = argument_parser()
    set_run_title(_params)
    _seed = _params["seed"]
    set_run_title(_params)
    is_mixed = True if _params["ma_type"] == 'mix_sum' else False
    # _seed = _params["seed"]
    for run_i in range(_params["nb_runs"]):

        _params["seed"] = _seed + run_i
        _meta_logger = meta_logger_initialization(_params["full_experiment_name"], _params['seed'])
        _params['daemon_fname'] = ''
        if _params['daemon']:
            _params['daemon_fname'] = f'{_params["experiment_filepath"]}/timestamp.log'
        print_out_configurations(_params)

        if _params["ma_type"] == 'maddpg':
            _train_env, _eval_env, _device, _params = make_env(_params)
            train_maddpg(_train_env, _eval_env, _meta_logger,
                         _params["max_step"], _params["evaluation_frequency"], _params['saving_frequency'],
                         _params["experiment_filepath"], _params["seed"],
                         _params['max_episode_len'], _params["learning_starts"],
                         save_model=_params["save_model"], daemon_fname=_params['daemon_fname'],
                         update_step=_params['update_step'], team_reward=_params['team_reward'])

        elif _params["ma_type"] == 'iddpg':
            _train_env, _eval_env, _device, _params = make_env(_params, discrete_action_space=False)
            # _device = 'cpu'
            # _agents = agent_networks_ddpg(_train_env, _params, _device)
            _agents = agent_networks_ddpg_2(_train_env, _params, _device)

            train_mpe_iddpg(_agents, _train_env, _eval_env, _meta_logger,
                            _params["max_step"], _params["evaluation_frequency"],
                            _params['saving_frequency'], _params["experiment_filepath"],
                            _params["seed"], _params['max_episode_len'], _params['learning_starts'],
                            _params["save_model"], daemon_fname=_params['daemon_fname'],
                            update_step=_params['update_step'])

        elif _params['ma_type'] == 'cddpg':
            _train_env, _eval_env, _device, _params = make_env(_params, discrete_action_space=False)
            _agent = agent_networks_ddpg_2_centralized(_train_env, _params, _device)
            train_mpe_cddpg(_agent, _train_env, _eval_env, _meta_logger,
                            _params["max_step"], _params["evaluation_frequency"],
                            _params['saving_frequency'], _params["experiment_filepath"],
                            _params["seed"], _params['max_episode_len'], _params['learning_starts'],
                            _params["save_model"], daemon_fname=_params['daemon_fname'],
                            update_step=_params['update_step'])

        else:
            pass


