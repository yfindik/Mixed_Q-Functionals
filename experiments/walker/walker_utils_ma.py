from time import time

import gym
import numpy as np
from pettingzoo.sisl import multiwalker_v9


def multiwalker_adjustment_centralized(org_env, obs_indexes=None):
    modified_env = lambda: None
    modified_env.org_env = org_env
    modified_env.reset = org_env.reset
    modified_env.step = org_env.step
    modified_env.obs_indexes = obs_indexes
    # mystery variable for now.
    # modified_env.action_space = org_env.action_space
    modified_env.action_spaces = org_env.action_spaces
    modified_env.possible_agents = list(modified_env.action_spaces.keys())
    modified_env.agents = list(modified_env.action_spaces.keys())
    modified_env.n = len(modified_env.agents)
    modified_env.observation_spaces = org_env.observation_spaces
    if obs_indexes:
        for i in modified_env.observation_spaces:
            modified_env.observation_spaces[i] = gym.spaces.Box(low=float(org_env.observation_spaces[i].low_repr),
                                                                high=float(org_env.observation_spaces[i].high_repr),
                                                                shape=(len(obs_indexes),),
                                                                dtype=np.float32)

    modified_env.observation_space = org_env.observation_space
    modified_env.action_space = org_env.action_space
    modified_env.obs_indexes = obs_indexes

    return modified_env


def multiwalker_step_centralized(modified_env, acts, act_mapped=False):

    data = modified_env.step(acts)

    observations, rewards, terminations, truncations, infos = data[0], data[1], data[2], data[3], data[4]
    if modified_env.obs_indexes:
        for key in observations:
            observations[key] = observations[key][modified_env.obs_indexes]

    observations = np.array(list(observations.values())).flatten()
    rewards = sum(rewards.values())
    terminations = all(terminations.values())
    truncations = all(truncations.values())
    return observations, rewards, terminations, truncations, infos


def multiwalker_reset_centralized(modified_env, seed=0):
    observations = modified_env.reset()[0]

    if modified_env.obs_indexes:
        for key in observations:
            observations[key] = observations[key][modified_env.obs_indexes]

    observations = np.array(list(observations.values())).flatten()
    return observations


def multiwalker_run_centralized(train_env, t=10):
    evaluation_rewards = []
    for i in range(t):
        obs_n, done = multiwalker_reset_centralized(train_env), False
        evaluation_reward = np.zeros(len(train_env.org_env.agents))
        max_r = 0
        min_r = 10
        while not done and train_env.org_env.agents:

            actions = {agent: train_env.action_space(agent).sample() for agent in train_env.agents}

            # environment step
            try:
                data = multiwalker_step_centralized(train_env, actions, act_mapped=True)
            except Exception as e:
                print(e)
            new_obs_n, rew_n, done_n, info_n = data[0], data[1], data[2], data[3]
            done = all(done_n.values())

            obs_n = new_obs_n
            if list(rew_n.values())[0] > max_r:
                max_r = list(rew_n.values())[0]

            if list(rew_n.values())[0] < min_r:
                min_r = list(rew_n.values())[0]

        evaluation_rewards.append(evaluation_reward)
        print(f'Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, Minimum Reward: {min_r}')


def multiwalker_adjustment_separate(org_env, obs_indexes=None):
    modified_env = lambda: None
    modified_env.org_env = org_env
    modified_env.reset = org_env.reset
    modified_env.step = org_env.step
    modified_env.obs_indexes = obs_indexes
    # mystery variable for now.
    # modified_env.action_space = org_env.action_space
    modified_env.action_spaces = {}
    for name, val in org_env.action_spaces.items():
        modified_env.action_spaces[f'{name}_joint_0'] = \
            modified_env.action_spaces[f'{name}_joint_1'] = \
            gym.spaces.Box(low=float(val.low_repr),
                           high=float(val.high_repr),
                           shape=(int(val.shape[0] / 2),),
                           dtype=np.float32)
    modified_env.possible_agents = list(modified_env.action_spaces.keys())
    modified_env.agents = list(modified_env.action_spaces.keys())
    modified_env.n = len(modified_env.agents)
    modified_env.nb_joints = 2
    modified_env.joint_indexes = {modified_env.possible_agents[0]: [4, 5, 6, 7, 8],
                                  modified_env.possible_agents[1]: [9, 10, 11, 12, 13]}

    agent_specific_obs = len(list(modified_env.joint_indexes.values())[0])
    modified_env.observation_spaces = {}

    for name, val in org_env.observation_spaces.items():
        if obs_indexes:
            new_obs_len = len(obs_indexes) - agent_specific_obs
        else:
            new_obs_len = val.shape[0] - agent_specific_obs

        modified_env.observation_spaces[f'{name}_joint_0'] = \
            modified_env.observation_spaces[f'{name}_joint_1'] = \
            gym.spaces.Box(low=float(val.low_repr),
                           high=float(val.high_repr),
                           shape=(new_obs_len,),
                           dtype=np.float32)

    modified_env.observation_space = org_env.observation_space
    modified_env.action_space = org_env.action_space
    modified_env.obs_indexes = obs_indexes
    return modified_env


def multiwalker_step_separate(modified_env, acts):
    action_dict = {}
    action_list = list(acts.values())

    for i, name in zip(range(0, len(acts), 2), modified_env.org_env.agents):
        action_dict[name] = list(np.concatenate((action_list[i], action_list[i + 1])))


    data = modified_env.step(action_dict)

    observations, rewards, terminations, truncations, infos = data[0], data[1], data[2], data[3], data[4]

    return render_data(observations, modified_env, rewards, terminations, truncations, infos, reset=False)


def multiwalker_reset_separate(modified_env, seed=0):
    observations = modified_env.reset()[0]
    return render_data(observations, modified_env)


def multiwalker_run_separate(train_env, t=10):
    evaluation_rewards = []
    for i in range(t):
        obs_n, done = multiwalker_reset_separate(train_env), False
        evaluation_reward = np.zeros(len(train_env.agents))
        max_r = 0
        min_r = 10
        while not done and train_env.org_env.agents:

            actions = {agent: train_env.action_spaces[agent].sample() for agent in train_env.agents}

            # environment step
            try:
                data = multiwalker_step_separate(train_env, actions)
            except Exception as e:
                print(e)
            new_obs_n, rew_n, done_n, info_n = data[0], data[1], data[2], data[3]
            done = all(done_n.values())

            obs_n = new_obs_n
            if list(rew_n.values())[0] > max_r:
                max_r = list(rew_n.values())[0]

            if list(rew_n.values())[0] < min_r:
                min_r = list(rew_n.values())[0]

        evaluation_rewards.append(evaluation_reward)
        print(f'Rewards: {evaluation_reward}, Team: {sum(evaluation_reward)}, Minimum Reward: {min_r}')


def render_data(observations, modified_env, rewards=None, terminations=None, truncations=None, infos=None, reset=True):
    modified_observations = {}
    modified_rewards = {}
    modified_terminations = {}
    modified_truncations = {}
    for i, key in enumerate(observations):
        for j in range(modified_env.nb_joints):
            agent_index = j + i * modified_env.nb_joints
            observation_indexes = modified_env.obs_indexes
            if not modified_env.obs_indexes:
                observation_indexes = list(range(observations[key].shape[0]))
                # removing other agents obs values
            modified_obs_indexes = np.delete(observation_indexes, list(modified_env.joint_indexes.values())[(j+1) % 2])
            modified_observations[modified_env.possible_agents[agent_index]] = observations[key][modified_obs_indexes]
            if not reset:
                modified_rewards[modified_env.possible_agents[agent_index]] = rewards[key]
                modified_terminations[modified_env.possible_agents[agent_index]] = terminations[key]
                modified_truncations[modified_env.possible_agents[agent_index]] = truncations[key]

    if reset:
        return modified_observations
    else:
        return modified_observations, modified_rewards, modified_terminations, modified_truncations, infos


if __name__ == "__main__":
    _train_env = multiwalker_v9.parallel_env(n_walkers=2, shared_reward=False,
                                             terrain_length=30, max_cycles=500,
                                             position_noise=0, angle_noise=0,
                                             forward_reward=10.0, render_mode='human')
    _eval_env = multiwalker_v9.parallel_env(n_walkers=2, shared_reward=False,
                                            terrain_length=30, max_cycles=500,
                                            position_noise=0, angle_noise=0,
                                            forward_reward=10.0, render_mode='human')

    # obs_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    obs_indexes = None

    e_env = multiwalker_adjustment_separate(_eval_env, obs_indexes=obs_indexes)
    multiwalker_run_separate(e_env)

    t_env = multiwalker_adjustment_centralized(_train_env)

    multiwalker_run_centralized(t_env)