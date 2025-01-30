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
import torch.nn.functional as F


from pettingzoo.sisl import multiwalker_v9

from functional_critic import utils, utils_for_q_learning
from functional_critic.agents_for_walkers import FourierAgent, LegendreAgent, PolynomialAgent
from logging_utils import MetaLogger

import datetime

from walker_utils_ma import multiwalker_adjustment_separate, multiwalker_reset_separate, \
    multiwalker_step_separate, multiwalker_adjustment_centralized, multiwalker_reset_centralized, \
    multiwalker_step_centralized


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

    parser.add_argument("--ma_type",
                        type=str,
                        help="Experiment Type",
                        required=True)

    parser.add_argument("--experiment_name",
                        type=str,
                        help="Experiment Name",
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

    parser.add_argument("--save_model", action="store_true")

    # sisl
    parser.add_argument("--is_gym_env", action='store_true')
    parser.add_argument("--is_mpe_env", action='store_true')
    parser.add_argument("--is_sisl_env", action='store_true')

    parser.add_argument("--terrain_length", default=30,
                        type=int)

    parser.add_argument("--n_walkers", default=1,
                        type=int)

    parser.add_argument("--forward_reward", default=5,
                        type=int)

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
#     params['is_not_gym_env'] = args.is_not_gym_env
    params["full_experiment_name"] = full_experiment_name
    params['evaluation_frequency'] = args.evaluation_frequency
    params['saving_frequency'] = args.saving_frequency
    params['max_episode_len'] = args.max_episode_len
    params['save_model'] = args.save_model or True
    params['experiment_filepath'] = os.path.join(os.getcwd(), os.path.join(args.experiment_name, args.run_title))
    params['experiment_name'] = args.experiment_name

    # sisl
    params['terrain_length'] = args.terrain_length
    params['n_walkers'] = args.n_walkers
    params['forward_reward'] = args.forward_reward
    params['is_gym_env'] = args.is_gym_env
    params['is_mpe_env'] = args.is_mpe_env
    params['is_sisl_env'] = args.is_sisl_env

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


def log_to_file(fname, text):
    with open(fname, "a") as fh:
       fh.write(str(datetime.datetime.now())+ ' ' + text + "\n")


def set_run_title(params):
    # params['max_episode']
    if params['ma_type'] == 'maddpg':
        t = 'team-reward_' if params['team_reward'] else 'individual-reward_'
        run_title = f"{params['ma_type']}_run_" \
                    f"{params['update_step']}-step-update_" \
                    f"{t}" \
                    f"nb-walkers-{params['n_walkers']}_" \
                    f"terrain-length-{params['terrain_length']}_" \
                    f"forward-reward-{params['forward_reward']}_" \
                    f"me-len-{params['max_episode_len']}_" \
                    f"episode-{params['max_episode']}_" \
                    f"lr{str(params['learning_rate']).replace('.', '')}_" \
                    f"tlr{str(params['target_network_learning_rate']).replace('.', '')}_" \
                    f"b{params['batch_size']}_l{params['num_layers']}_" \
                    f"g{str(params['gamma']).replace('.', '')}_" \
                    f"n{params['layer_size']}_m{params['max_buffer_size']}"
    else:
        run_title = f"{params['ma_type']}_run_" \
                    f"{params['update_step']}-step-update_" \
                    f"nb-walkers-{params['n_walkers']}_" \
                    f"terrain-length-{params['terrain_length']}_" \
                    f"forward-reward-{params['forward_reward']}_" \
                    f"me-len-{params['max_episode_len']}_" \
                    f"func-{params['functional']}_rank-{params['rank']}_episode-{params['max_episode']}_" \
                    f"lr{str(params['learning_rate']).replace('.', '')}_" \
                    f"b{params['batch_size']}_l{params['num_layers']}_" \
                    f"g{str(params['gamma']).replace('.', '')}_" \
                    f"n{params['layer_size']}_m{params['max_buffer_size']}"

    full_experiment_name = os.path.join(params['experiment_name'], run_title)
    utils.create_log_dir(full_experiment_name)
    hyperparams_dir = utils.create_log_dir(
        os.path.join(full_experiment_name, "hyperparams"))

    params["full_experiment_name"] = full_experiment_name
    params["hyperparams_dir"] = hyperparams_dir
    params['experiment_filepath'] = os.path.join(os.getcwd(), os.path.join(params['experiment_name'], run_title))


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


def update_with_mixing(Q_objects, Q_targets, optimizer, params, gamma, step=-1, grad_clip_norm=5):
    update_params = []
    # update_param = {"average_Q": 0, "average_Q_star": 0, "fourier_coefficients": None}
    buffer_problem_flag = False
    for agent_index, (Q_object, Q_target) in enumerate(list(zip(Q_objects, Q_targets))):
        if len(Q_object.buffer_object) < Q_object.params['batch_size']:
            buffer_problem_flag = True
        update_params.append({"average_Q": 0, "average_Q_star": 0, "fourier_coefficients": None})

    if buffer_problem_flag:
        return 0, update_params
    loss = 0
    for _ in range(1):

        Qs = []
        Q_stars = []
        Rs = []
        Dones = []
        for agent_index, (Q_object, Q_target) in enumerate(list(zip(Q_objects, Q_targets))):
            s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = Q_object.buffer_object.sample(
                Q_object.params['batch_size'])
            # clip rewards
            r_matrix = np.clip(r_matrix, a_min=-Q_object.params['reward_clip'], a_max=Q_object.params['reward_clip'])

            s_matrix = torch.from_numpy(s_matrix).float().to(Q_object.device)
            a_matrix = torch.from_numpy(a_matrix).float().to(Q_object.device)
            r_matrix = torch.from_numpy(r_matrix).float().to(Q_object.device)
            done_matrix = torch.from_numpy(done_matrix).float().to(Q_object.device)
            sp_matrix = torch.from_numpy(sp_matrix).float().to(Q_object.device)

            Q = Q_object.forward(s_matrix, a_matrix)

            Q_star = Q_target.compute_Q_star(sp_matrix,
                                             quantile_sampling=Q_object.params['use_quantile_sampling_bootstrapping'],
                                             step=step, do_minq=Q_object.params['minq'])

            Q_star = Q_star.reshape((Q_object.params['batch_size'], -1))

            Qs.append(Q)
            Q_stars.append(Q_star)
            Rs.append(r_matrix)
            Dones.append(done_matrix)

            update_params[agent_index]["average_Q_star"] = Q_star.mean().cpu().item()
            update_params[agent_index]["average_Q"] = Q.mean().cpu().item()

        Qs_tensor = torch.cat(Qs, axis=1)
        Q_stars_tensor = torch.cat(Q_stars, axis=1)
        Rs_tensor = torch.cat(Rs, axis=1)
        Dones_tensor = torch.cat(Dones, axis=1)

        Q_total = Qs_tensor.sum(dim=1, keepdims=True)
        Q_star_total = Q_stars_tensor.sum(dim=1, keepdims=True)
        R_total = Rs_tensor.sum(dim=1, keepdims=True)
        # R_total = torch.matmul(Rs_tensor, torch.FloatTensor([[1, 0], [0, 1]]).to('cuda')).sum(dim=1, keepdims=True)
        Done_total = torch.all(Dones_tensor, dim=1, keepdims=True)

        y = R_total + gamma * Q_star_total * (1 - Done_total.long())

        # since the criterion same for all agent we are using one of Q_objects
        # loss = Q_object.criterion(Q_total, y.detach())

        loss += F.smooth_l1_loss(Q_total, y.detach())

    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(chain.from_iterable([param['params'] for param in params])
    #                                , grad_clip_norm, norm_type=2)
    optimizer.step()
    optimizer.zero_grad()

    for agent_index, (Q_object, Q_target) in enumerate(list(zip(Q_objects, Q_targets))):
        utils_for_q_learning.sync_networks(
            target=Q_target,
            online=Q_object,
            alpha=Q_object.params['target_network_learning_rate'],
            copy=False)

    loss_data = loss.cpu().item()
    return loss_data, update_params


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


def agent_buffer_initialization(train_env, Q_objects, warm_up_steps):
    print("Start the Initialization Process!")
    steps = 0
    while steps < warm_up_steps:
        s, done = multiwalker_reset_separate(train_env), False

        while not done and train_env.org_env.agents:

            actions = {agent: train_env.action_spaces[agent].sample() for agent in train_env.agents}

            # environment step
            try:
                data = multiwalker_step_separate(train_env, actions)
            except Exception as e:
                print(e)
            s_, r, done, info = data[0], data[1], data[2], data[3]

            for agent_index, agent_name in enumerate(train_env.agents):
                Q_objects[agent_index].buffer_object.append(s[agent_name], actions[agent_name],
                                                            r[agent_name], done[agent_name], s_[agent_name])

            s = s_
            steps += 1
            done = all(done.values())

    print("The initialization process finished!")


def train(Q_objects, Q_targets, train_env, eval_env, meta_logger,
          max_step, policy_type, evaluation_frequency, saving_frequency,
          experiment_filepath, seed, max_episode_len,
          save_model=True, is_mixed=False, daemon_fname='', update_step=50):
    if is_mixed:
        params = []
        for Q_object in Q_objects:
            params += Q_object.params_dic
        common_optimizer = torch.optim.Adam(params)

    total_steps = 0
    episodes = 0
    # epsilons = []
    per_step_losses = []
    qs = []
    q_stars = []
    episodic_rewards = []
    iter_start_time = time()
    while total_steps < max_step:
        s, done = multiwalker_reset_separate(train_env), False

        episodic_reward = np.zeros(len(Q_objects))
        steps = 0
        while not done and train_env.org_env.agents:
            actions = {}
            for agent_index, agent_name in enumerate(train_env.agents):
                actions[agent_name] = list(Q_objects[agent_index].enact_policy(s[agent_name], episodes + 1, steps, 'train', policy_type))

            try:
                data = multiwalker_step_separate(train_env, actions)
            except Exception as e:
                print(e)
            s_, r, done, info = data[0], data[1], data[2], data[3]

            for agent_index, agent_name in enumerate(train_env.agents):
                Q_objects[agent_index].buffer_object.append(s[agent_name], actions[agent_name],
                                                            r[agent_name], done[agent_name], s_[agent_name])

            s = s_

            if total_steps % update_step == 0:

                losses = []
                basis_statistics = []
                agents_q = []
                agents_q_star = []
                if is_mixed:
                    loss, b = update_with_mixing(Q_objects, Q_targets, common_optimizer, params, 0.99, step=steps)
                    for basis_statistic in b:
                        losses.append(loss)
                        basis_statistics.append(basis_statistic)
                        agents_q.append(basis_statistic["average_Q"])
                        agents_q_star.append(basis_statistic["average_Q_star"])

                else:
                    # independent q-functionals
                    for agent_index, Q_object in enumerate(Q_objects):
                        loss, basis_statistic = Q_object.update(Q_targets[agent_index], step=steps)
                        losses.append(loss)
                        basis_statistics.append(basis_statistic)
                        agents_q.append(basis_statistic["average_Q"])
                        agents_q_star.append(basis_statistic["average_Q_star"])

                per_step_losses.append(losses)
                qs.append(agents_q)
                q_stars.append(agents_q_star)

            if (total_steps % evaluation_frequency == 0) or (total_steps == max_step - 1):
                evaluation_rewards = []
                kk = False
                for _ in range(10):
                    evaluation_reward = np.zeros(eval_env.n)
                    s_eval, done_eval = multiwalker_reset_separate(eval_env), False
                    steps_eval = 0
                    while not done_eval and eval_env.org_env.agents:
                        actions_eval = {}
                        for agent_index, agent_name in enumerate(eval_env.agents):
                            actions_eval[agent_name] = list(Q_objects[agent_index].e_greedy_policy(s_eval[agent_name],
                                                                                                   episodes + 1, steps,
                                                                                                   'test'))

                        # environment step
                        try:
                            data_eval = multiwalker_step_separate(eval_env, actions_eval)
                        except Exception as e:
                            print(e)
                        s_eval_, r_eval, done_eval = data_eval[0], data_eval[1], data_eval[2]

                        try:
                            evaluation_reward += np.array(list(r_eval.values())[:eval_env.n])
                        except Exception as e:
                            print(e)
                        s_eval = s_eval_

                        if kk:
                            eval_env.render()

                        steps_eval += 1
                        done_eval = all(done_eval.values())

                    evaluation_rewards.append(evaluation_reward)

                if daemon_fname != '':
                    log_to_file(daemon_fname, f"Episodes {episodes}:, ")
                    log_to_file(daemon_fname, f"Total steps {total_steps}/{max_step}, ")
                    log_to_file(daemon_fname, f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)},")
                    log_to_file(daemon_fname, f"Time: {round(time() - iter_start_time, 3)}")
                else:
                    print(f"Episodes {episodes}:, "
                          f"Total steps {total_steps}/{max_step}, "
                          f"Evaluation Reward: {np.mean(evaluation_rewards, axis=0)}," # mean per agent
                          f"Time: {round(time() - iter_start_time, 3)}")

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

            if save_model and ((total_steps % saving_frequency == 0) or total_steps == (max_step - 1)):
                path = os.path.join(experiment_filepath, "logs")
                if not os.path.exists(path):
                    try:
                        os.makedirs(path, exist_ok=True)
                    except OSError:
                        print("Creation of the directory %s failed" % path)
                    else:
                        print("Successfully created the directory %s " % path)
                for agent_index, (Q_object, Q_target) in enumerate(list(zip(Q_objects, Q_targets))):
                    torch.save(Q_object.state_dict(), os.path.join(path, f"seed_{seed}_object_{agent_index}_steps_" +
                                                                   str(total_steps + 1)))
                    torch.save(Q_target.state_dict(), os.path.join(path, f"seed_{seed}_target_{agent_index}_steps_" +
                                                                   str(total_steps + 1)))

            steps += 1
            total_steps += 1
            try:
                episodic_reward += np.array(list(r.values())[:train_env.n])
            except Exception as e:
                print(e)
            done = all(done.values())

        episodes += 1
        # epsilons.append(eps)
        episodic_rewards.append(episodic_reward)


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


def train_centralized(Q_object, Q_target, train_env, eval_env, meta_logger,
                      max_step, policy_type,
                      evaluation_frequency, saving_frequency,
                      experiment_filepath, seed, max_episode_len,
                      save_model=True, daemon_fname='', update_step=50):

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
        s, done = multiwalker_reset_centralized(train_env), False

        episodic_reward = np.zeros(train_env.n)
        steps = 0
        while not done and train_env.org_env.agents:

            fp_t = time()

            act = Q_object.enact_policy(s, episodes + 1, steps, 'train', policy_type)

            actions = {}
            for agent_name, action in zip(train_env.agents, np.array_split(act, train_env.n)):
                actions[agent_name] = list(action)

            try:
                data = multiwalker_step_centralized(train_env, actions)
            except Exception as e:
                print(e)
            s_, r, done, info = data[0], data[1], data[2], data[3]

            try:
                Q_object.buffer_object.append(s, act, r,  done, s_)
            except Exception as e:
                print(e)
            fp += time() - fp_t

            s = s_

            losses = []
            basis_statistics = []
            agents_q = []
            agents_q_star = []

            sp_t = time()

            if total_steps % update_step == 0:  # only update every 100 steps
                loss, basis_statistic = Q_object.update(Q_target, step=steps)
                losses.append(loss)
                basis_statistics.append(basis_statistic)
                agents_q.append(basis_statistic["average_Q"])
                agents_q_star.append(basis_statistic["average_Q_star"])

            sp += time() - sp_t

            per_step_losses.append(losses)
            qs.append(agents_q)
            q_stars.append(agents_q_star)

            if (total_steps % evaluation_frequency == 0) or (total_steps == max_step - 1):
                evaluation_rewards = []
                kk = False
                for _ in range(10):
                    evaluation_reward = np.zeros(eval_env.n)
                    s_eval, done_eval = multiwalker_reset_centralized(eval_env), False

                    steps_eval = 0
                    while not done_eval and eval_env.org_env.agents:

                        act_eval = Q_object.e_greedy_policy(s_eval, 1, steps_eval, 'test')

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
                torch.save(Q_object.state_dict(), os.path.join(path, f"seed_{seed}_object_steps_" +
                                                               str(total_steps + 1)))
                torch.save(Q_target.state_dict(), os.path.join(path, f"seed_{seed}_target_steps_" +
                                                                   str(total_steps + 1)))

            steps += 1
            total_steps += 1
            episodic_reward += r

        episodes += 1
        # epsilons.append(eps)
        episodic_rewards.append(episodic_reward)



if __name__ == "__main__":


    _params = argument_parser()
    _seed = _params["seed"]

    set_run_title(_params)
    _params['obs_indexes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 30]

    for run_i in range(_params["nb_runs"]):
        # if run_i < 2:
        #     continue
        _params["seed"] = _seed + run_i
        _meta_logger = meta_logger_initialization(_params["full_experiment_name"], _params['seed'])
        _params['daemon_fname'] = ''
        if _params['daemon']:
            _params['daemon_fname'] = f'{_params["experiment_filepath"]}/timestamp.log'
        print_out_configurations(_params)


        if _params["ma_type"] == 'mix_sum' or _params["ma_type"] == 'ind':
            if _params["is_sisl_env"]:
                _train_env, _eval_env, _device, _params = make_walker(_params, adj=True, centralized=False)
                _Q_objects, _Q_targets = agent_networks(_params, _train_env, _device)
                agent_buffer_initialization(_train_env, _Q_objects, _params["learning_starts"])
                is_mixed = True if _params["ma_type"] == 'mix_sum' else False
                train(_Q_objects, _Q_targets, _train_env, _eval_env, _meta_logger,
                      _params["max_step"],
                      _params["policy_type"],
                      _params["evaluation_frequency"],
                      _params["saving_frequency"],
                      _params["experiment_filepath"], _params["seed"], _params['max_episode_len'],
                      _params["save_model"], is_mixed=is_mixed,
                      update_step=_params['update_step'], daemon_fname=_params['daemon_fname'])


        elif _params["ma_type"] == 'centralized':
            if _params["is_sisl_env"]:
                _train_env, _eval_env, _device, _params = make_walker(_params, adj=True, centralized=True)
                _Q_object, _Q_target = agent_networks_centralized(_params, _train_env, _device)
                train_centralized(_Q_object, _Q_target, _train_env, _eval_env, _meta_logger,
                                  _params["max_step"], _params["policy_type"], _params["evaluation_frequency"],
                                  _params['saving_frequency'],
                                  _params["experiment_filepath"], _params["seed"], _params['max_episode_len'],
                                  _params["save_model"], daemon_fname=_params['daemon_fname'],
                                  update_step=_params['update_step'])


