import random
from functional_critic import utils, utils_for_q_learning
from functional_critic.utils import cartesian_sum, buffer_class
import torch
import torch.nn as nn
import numpy as np
from numpy.polynomial import Polynomial
import gym

class BaseAgent(nn.Module):
    """
    The base agent for all various basis
    """
    num_coefficient_scale = 1  # The scale factor for different basis

    def __init__(self, params, env, device, seed, centralized=False):
        super(BaseAgent, self).__init__()

        self.env = env

        if not centralized:
            if params['is_not_gym_env'] and env.discrete_action_space:
                self.is_env_discrete = False
                self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.env.action_space[0].n,),
                                                   dtype=np.float32)
                self.state_space = self.env.observation_space[0]

            elif params['is_not_gym_env']:
                self.is_env_discrete = False
                self.action_space = self.env.action_space[0]
                self.state_space = self.env.observation_space[0]

            else:
                if type(self.env.action_space) == gym.spaces.Discrete:
                    self.is_env_discrete = True
                    # pd for each action
                    self.action_space = gym.spaces.Box(low=0.0, high=1, shape=(self.env.action_space.n,),
                                                       dtype=np.float32)
                    self.state_space = self.env.observation_space
                if isinstance(self.env.action_space, list):
                    self.is_env_discrete = True
                    # pd for each action
                    self.action_space = gym.spaces.Box(low=0.0, high=1, shape=(self.env.action_space[0].n,),
                                                       dtype=np.float32)
                    self.state_space = self.env.observation_space[0]

                else:
                    self.is_env_discrete = False
                    self.action_space = self.env.action_space
                    self.state_space = self.env.observation_space

        else:
            if params['is_not_gym_env'] and env.discrete_action_space:
                return
            elif params['is_not_gym_env']:
                self.is_env_discrete = False
                self.action_space = gym.spaces.Box(low=-1.0, high=1.0,
                                                   shape=(self.env.action_space[0].shape[0] * self.env.n,),
                                                   dtype=np.float32)
                self.state_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                  shape=(self.env.observation_space[0].shape[0] * self.env.n,),
                                                  dtype=np.float32)
            else:
                return

        self.device = device
        self.params = params
        self.state_size = self.state_space.shape[0]
        self.action_size = self.action_space.shape[0]

        self.max_a = self.action_space.high[0]
        self.min_a = self.action_space.low[0]
        assert np.allclose(self.action_space.high, self.max_a)

        self.act_dim = self.action_size

        self.L = (self.max_a - self.min_a)

        self.rank = self.params['rank']
        self.num_hidden_layers = self.params['num_layers']

        self.inner_arg_freq = torch.from_numpy(np.array(cartesian_sum(self.rank, self.act_dim)))
        self.inner_arg_freq = self.inner_arg_freq.reshape(-1, self.act_dim).to(self.device)  # [num_pairs, act_dim]

        if self.params["coefficient_scaling_exponent"] != 0:
            inverse_norm_scaling = torch.transpose(self.inner_arg_freq.unsqueeze(dim=0).to(torch.float32), 0, 1).norm(
                dim=[1, 2])
            inverse_norm_scaling[inverse_norm_scaling == 0] = 1  # replace 0 values with 1 to avoid division by 0 errors
            inverse_norm_scaling = torch.pow(inverse_norm_scaling, self.params["coefficient_scaling_exponent"])
            inverse_norm_scaling = 1 / inverse_norm_scaling
            inverse_norm_scaling = inverse_norm_scaling.repeat(1, type(self).num_coefficient_scale).to(self.device)
            self.inverse_norm_scaling = inverse_norm_scaling

        self.network_output_dims = type(self).num_coefficient_scale * self.inner_arg_freq.shape[0]
        print("network output dims:", self.network_output_dims)

        self.buffer_object = buffer_class(
            max_length=self.params['max_buffer_size'],
            env=self.env,
            seed_number=seed,
            centralized=centralized)

        self.action_to_sample = params["qstar_samples"]
        self.interaction_action_to_sample = params.get("interaction_qstar_samples", self.action_to_sample)

        # self.visualization_state = torch.from_numpy(self.state_space.sample()).to(self.device)

        # Quantile schedule is an actual function now, since it got complicated. Takes into account anneal/not

        assert not (params.get("sample_method") == 'uniform' and params.get("anneal_std", False)), \
            "Illegal combination of uniform sampling and std-annealing"

        assert not (params.get("use_quantile_sampling_bootstrapping") and params.get("use_target_policy_smoothing")), \
            "Illegal combination of quantile sampling and target policy smoothing"

        """
        Define the network structure
        """
        final_layer_init_scale = params.get("final_layer_init_scale", 1.0)

        activation_str = params.get("activation")
        if activation_str.lower() == "relu":
            activation = nn.ReLU
        elif activation_str.lower() == "tanh":
            activation = nn.Tanh
        else:
            raise Exception("Unrecognized activation function: {}".format(activation_str))

        hidden_1 = []
        for _ in range(self.num_hidden_layers):
            hidden_1.append(nn.Linear(self.params['layer_size'], self.params['layer_size']))
            hidden_1.append(activation())

        self._coefficient_module_1 = nn.Sequential(
            nn.Linear(self.state_size, self.params['layer_size']),
            activation(),
            *hidden_1,
            nn.Linear(self.params['layer_size'], self.network_output_dims),
        )
        with torch.no_grad():
            self._coefficient_module_1[-1].weight.data = self._coefficient_module_1[
                                                             -1].weight.data * final_layer_init_scale
            self._coefficient_module_1[-1].bias.data = self._coefficient_module_1[-1].bias.data * final_layer_init_scale

        # have another network if using the TD3 trick
        if self.params['minq']:
            hidden_2 = []
            for _ in range(self.num_hidden_layers):
                hidden_2.append(nn.Linear(self.params['layer_size'], self.params['layer_size']))
                hidden_2.append(activation())

            self._coefficient_module_2 = nn.Sequential(
                nn.Linear(self.state_size, self.params['layer_size']),
                activation(),
                *hidden_2,
                nn.Linear(self.params['layer_size'], self.network_output_dims),
            )
            with torch.no_grad():
                self._coefficient_module_2[-1].weight.data = self._coefficient_module_2[
                                                                 -1].weight.data * final_layer_init_scale
                self._coefficient_module_2[-1].bias.data = self._coefficient_module_2[
                                                               -1].bias.data * final_layer_init_scale

        """
        Define the optimizer
        """
        if self.params['loss_type'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif self.params['loss_type'] == 'HuberLoss':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise NameError('only two kinds of loss can we use, MSELoss or HuberLoss')

        self.params_dic = [{
            'params': self._coefficient_module_1.parameters(),
            'lr': self.params['learning_rate']
        }]

        if self.params['minq']:
            self.params_dic.append({
                'params': self._coefficient_module_2.parameters(),
                'lr': self.params['learning_rate']
            })

        if (params.get('q_optimizer', 'Adam') == 'SGD'):
            print("using SGD for q func optimizer in coefficient net")
            self.optimizer = torch.optim.SGD(self.params_dic, lr=3e-4)
        else:
            print("using adam")
            self.optimizer = torch.optim.Adam(self.params_dic)

        self.T = params['entropy_regularized']

        """
        Define the precomputed actions and representations
        """
        self.use_precomputed_basis = self.params.get('use_precomputed_basis', False)
        if self.use_precomputed_basis:
            with torch.no_grad():
                self._num_precomputed = self.params.get('num_precomputed', 1000000)
                precomputed_actions, precomputed_basis_representations = self._construct_precomputed_basis(
                    self._num_precomputed)
            self.precomputed_actions = precomputed_actions
            self.precomputed_basis_representations = precomputed_basis_representations

        self.to(self.device)

    def _construct_precomputed_basis(self, num_precomputed, chunk_size=1000):
        """The basis computation involves a lot of intermediate computation,
        which can make you run out of memory during construction even when the final tensors aren't that large.
        This splits the computation into chunks so that the intermediate tensors get discarded.
        """
        assert chunk_size > 0
        action_chunks = []
        representation_chunks = []
        end_index = 0
        while end_index != num_precomputed:
            start_index = end_index
            end_index = min(start_index + chunk_size, num_precomputed)
            actions = torch.rand((end_index - start_index, self.act_dim)) * (2 * self.max_a) - self.max_a
            actions = actions.to(self.device)
            actions, representations = self.get_actions_and_basis_representations(
                actions, prefetch_basis_shape=None)
            action_chunks.append(actions)
            representation_chunks.append(representations)
        precomputed_actions = torch.cat(action_chunks, dim=0).to(self.device)
        precomputed_basis_representations = torch.cat(representation_chunks, dim=0).to(self.device)
        return precomputed_actions, precomputed_basis_representations

    def quantile_schedule(self, step=-1):
        if self.params['anneal_quantile_sampling']:
            return self.params['quantile_sampling_percent'] - (
                    self.params['quantile_sampling_percent'] * step / self.params['max_step'])
        else:
            return self.params['quantile_sampling_percent']

    def std_gaussian_schedule(self, step=-1):
        if self.params['anneal_std'] and (step != -1):
            return self.max_a * (step / self.params['max_step'])
        else:
            return self.max_a

    def get_actions_and_basis_representations(self, actions=None, prefetch_basis_shape=None):
        """
        Implement this function in each basis class
        """
        raise NotImplementedError("Shoule have a solid implementation for calculating the action representations.")

    def uniformly_sample_action_space(self, action_to_sample=None):
        if action_to_sample is None:
            action_to_sample = self.action_to_sample
        # return torch.rand((action_to_sample, self.act_dim), device=self.device) * (2 * self.max_a) - self.max_a

        sampled_actions = torch.FloatTensor(action_to_sample, self.act_dim)\
            .uniform_(self.min_a, self.max_a).to(self.device)

        return sampled_actions
        # return sampled_actions.floor() if self.is_env_discrete else sampled_actions

    def gaussian_sample_action_space(self, step, action_to_sample=None):
        if action_to_sample is None:
            action_to_sample = self.action_to_sample
        return torch.normal(0, self.std_gaussian_schedule(step) + 1e-4, (action_to_sample, self.act_dim),
                            device=self.device)

    def sample_action_space(self, step, action_to_sample=None):
        if self.params['sample_method'] == 'uniform':
            return self.uniformly_sample_action_space(action_to_sample=action_to_sample)
        elif self.params['sample_method'] == 'gaussian':
            return self.gaussian_sample_action_space(step=(step if self.params['anneal_std'] else -1),
                                                     action_to_sample=action_to_sample)
        else:
            raise Exception(f"expected uniform or gaussian, got {self.params['sample_method']}")

    def get_best_qvalue_and_action(self, s, step=-1, action_to_sample=None):
        """
        given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
        :param s:
        :param step:
        :param action_to_sample:
        :return:
        """
        if action_to_sample is None:
            action_to_sample = self.action_to_sample

        should_quantile_sample = True
        if step == -1:  # set step = -1 during evaluation/training if you don't want to use quantile sampling
            should_quantile_sample = False

        best_action, best_q = self.compute_Q_star(s, return_best_actions=True, quantile_sampling=should_quantile_sample,
                                                  step=step, action_to_sample=action_to_sample)
        if s.shape[0] == 1:
            best_action = best_action.squeeze(dim=0)
            best_q = best_q.squeeze(dim=0)

        return best_q, best_action.cpu()

    def get_all_q_values_and_action_set(self, s, actions=None, prefetch_basis_shape=None):
        """
        Run through n actions in each state, and choose the max to determine what the real Q* value is.
        :param s:
        :param actions:
        :param prefetch_basis_shape:
        :return:
        """
        with torch.no_grad():
            coefficients = self.get_coefficients(self._coefficient_module_1,
                                                 s)  # [batch_size, num_coefficient_scale * inner_arg_freq]
            _, action_representations = self.get_actions_and_basis_representations(actions,
                                                                                   prefetch_basis_shape=prefetch_basis_shape)
            coefficients = coefficients.unsqueeze(dim=1)  # [256 x 1 x 50]
            output = action_representations.mul(
                coefficients)  # [256 x 500 x 50] --> [batch_dim x num_actions_to_sample x self.rank**self.act_dim]
            output = output.sum(dim=2)  # [batch_dim x num_actions_to_sample]

            return output

    def forward(self, s, a, coefficients=None):
        """
        given a batch of s,a , compute Q(s,a) [batch x 1]
        :param s: [batch_dim x self.state_size]
        :param a: [batch_dim x self.act_dim]
        :param coefficients:
        :return:
        """

        if coefficients is None:
            coefficients = self.get_coefficients(self._coefficient_module_1, s)

        _, action_representations = self.get_actions_and_basis_representations(actions=a)
        output = action_representations.mul(coefficients).sum(dim=1, keepdim=True)  # [batch, 1]

        return output

    def compute_Q_star(self, s, return_best_actions=False, quantile_sampling=False, step=-1, do_minq=False,
                       action_to_sample=None):
        if action_to_sample is None:
            action_to_sample = self.action_to_sample

        with torch.no_grad():
            coefficients_1 = self.get_coefficients(self._coefficient_module_1, s)

            # only use minq if self.params['minq'] and we are forming the bellman target in update. 
            if do_minq:
                coefficients_2 = self.get_coefficients(self._coefficient_module_2, s)

            if self.use_precomputed_basis:
                prefetch_basis_shape = [s.shape[0], action_to_sample]
                actions = None
            else:
                prefetch_basis_shape = None
                actions = self.sample_action_space(step=step, action_to_sample=action_to_sample)
                actions = actions.unsqueeze(dim=0)  # [1 x 500 x 1]
                actions = actions.repeat(s.shape[0], 1, 1)  # [batch_num x 500 x 1]

            actions, action_representations = self.get_actions_and_basis_representations(actions=actions,
                                                                                         prefetch_basis_shape=prefetch_basis_shape)

            coefficients_1 = coefficients_1.unsqueeze(dim=1)  # [256 x 1 x 50]
            output = action_representations.mul(
                coefficients_1)  # [256 x 500 x 50] --> [batch_dim x num_actions_to_sample x self.rank**self.act_dim]

            # Use repeated torch.sum to reduce
            if do_minq:
                output = output.sum(dim=2)
                coefficients_2 = coefficients_2.unsqueeze(dim=1)
                output_2 = action_representations.mul(coefficients_2)
                output_2 = output_2.sum(dim=2)
                catted_action_values = torch.cat((output.unsqueeze(-1), output_2.unsqueeze(-1)),
                                                 dim=2)  # [batch x num_sampled_actions x 2]
                output = torch.min(catted_action_values, dim=2).values
            else:
                output = output.sum(dim=2)  # [batch_dim x num_actions_to_sample]

            if self.params['entropy_regularized'] > 0:
                # use entropy regularization to construct the bellman target.
                # otherwise, choose the best action using a softmax policy
                if output.shape[0] > 1:
                    # this is the case where we are calling from update(), asking for q star values back
                    output = self.T * torch.logsumexp(output / self.T, dim=1) - self.T * torch.log(
                        torch.tensor([action_to_sample]).to(torch.float32).to(self.device))
                    return output.unsqueeze(dim=1)  # [batch x 1]
                else:
                    # this is the case where we are interacting with the environment (ie, batch is 1) and we need to softmax. 
                    soft_fn = torch.nn.Softmax(dim=1)
                    with torch.no_grad():
                        softmaxed_output = soft_fn(output / self.T)
                        # sample from softmax distribution over action values 
                        index_sample = torch.multinomial(softmaxed_output, 1)
                    action_sampled = actions[0, index_sample.item(), :]
                    return action_sampled.reshape(1, -1), output[0, index_sample.item()]

            if quantile_sampling and self.quantile_schedule(step) != 0:
                QUANTILE = self.quantile_schedule(step)  # self.params['quantile_sampling_percent']
                assert QUANTILE > 0 and QUANTILE <= self.params["quantile_sampling_percent"], QUANTILE
                topn = max(1, int(QUANTILE * action_to_sample))
                quantiles = torch.topk(input=output, dim=1, k=topn)

                quantile_values = quantiles.values
                quantile_indexes = quantiles.indices

                # in the case where we are performing an update over a batch return the mean over the quantile for better stability
                if s.shape[0] != 1:
                    return torch.mean(quantile_values, dim=1)

                random_index = torch.randint(low=0, high=topn, size=(s.shape[0], 1)).to(self.device)
                best_indices = torch.gather(quantile_indexes, 1, random_index)
                output = torch.gather(output, 1, best_indices)
                best_actions = torch.gather(actions, 1,
                                            best_indices.unsqueeze(dim=2).repeat(1, 1, self.act_dim)).reshape(-1,
                                                                                                              self.act_dim)

            else:
                # this represents a tensor with num_actions_to_sample Q values, we just need to pick the best one now!
                output = torch.max(output, 1)

                best_actions = torch.gather(actions, 1,
                                            output.indices.reshape(-1, 1, 1).repeat(1, 1, self.act_dim)).reshape(-1,
                                                                                                                 self.act_dim)
                output = output.values.unsqueeze(dim=1)  # [batch_dim x 1]
                if self.params['use_target_policy_smoothing']:
                    # First, get noisy actions
                    smoothing_scale = self.params['tps_scale']
                    noise_max = smoothing_scale * 2.5
                    num_to_smooth_over = self.params['num_to_smooth_over']
                    additive_noise = torch.randn(s.shape[0], num_to_smooth_over, self.act_dim, device=self.device)
                    additive_noise = torch.clip(additive_noise * smoothing_scale, min=-noise_max, max=noise_max)
                    best_actions_expanded = best_actions.unsqueeze(1).repeat(1, num_to_smooth_over, 1)
                    noisy_actions = torch.clip(best_actions_expanded + additive_noise,
                                               min=-1 * self.max_a, max=self.max_a)
                    # Then, get all of their representations and values
                    _, noisy_action_representations = self.get_actions_and_basis_representations(actions=noisy_actions)
                    noisy_outputs = noisy_action_representations.mul(coefficients_1)
                    noisy_outputs = noisy_outputs.sum(dim=2)
                    if do_minq:
                        noisy_outputs_2 = noisy_action_representations.mul(coefficients_2)
                        noisy_outputs_2 = noisy_outputs_2.sum(dim=2)
                        catted_action_values = torch.cat((noisy_outputs.unsqueeze(-1), noisy_outputs_2.unsqueeze(-1)),
                                                         dim=2)
                        noisy_outputs = torch.min(catted_action_values, dim=2).values
                    # Finally, smooth over noisy outputs. Replaces output value
                    output = torch.mean(noisy_outputs, 1)  # mean over noisy outputs.

            if (s.shape[0] == 1 and return_best_actions):
                return best_actions, output
            else:
                return output

    def get_coefficients(self, module, states):
        coefficients = module(states)
        if self.params["coefficient_scaling_exponent"] != 0:
            coefficients = coefficients.mul(self.inverse_norm_scaling)
        return coefficients

    def get_coefficient_statistics(self, states):
        coefficients = self.get_coefficients(self._coefficient_module_1, states)
        bias_index = self.inner_arg_freq.shape[0]

        return dict(
            coefficients_mean=torch.mean(torch.mean(coefficients, dim=1)).item(),
            coefficients_max=torch.mean(torch.max(coefficients, dim=1).values).item(),
            coefficients_min=torch.mean(torch.min(coefficients, dim=1).values).item(),
            coefficients_bias_term=torch.mean(coefficients[:, bias_index]).item(),
            coefficients_bias_term_raw=coefficients[:, bias_index].cpu().detach().numpy(),
            coefficients_high_order_term=torch.mean(coefficients[:, -1]).item(),
        )

    def _use_quantile_sampling(self, train_or_test):
        assert train_or_test in ("train", "test")
        if train_or_test == "train":
            return self.params['use_quantile_sampling_training_interaction']
        else:
            return self.params['use_quantile_sampling_evaluation_interaction']

    def enact_policy(self, s, episode, step, train_or_test, policy_type="e_greedy"):
        assert policy_type in ["e_greedy", "e_greedy_gaussian", "gaussian",
                               "softmax"], f"Bad policy type: {policy_type}"
        policy_types = {
            'e_greedy': self.e_greedy_policy,
            'gaussian': self.gaussian_policy,
        }

        return policy_types[policy_type](s, episode, step, train_or_test)

    def e_greedy_policy(self, s, episode, step, train_or_test):
        """
        Given state s, at episode, take random action with p=eps if training Note - epsilon is determined by episode
        :param s:
        :param episode:
        :param step:
        :param train_or_test:
        :return:
        """
        assert train_or_test in ("train", "test")
        action_to_sample = self.interaction_action_to_sample
        # epsilon = max(0.01, 1 - (1 - 0.01) * (episode / (0.8 * 2000)))
        epsilon = 1 / np.power(episode,
                                  1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.action_space.sample()
            return a
        else:

            self.eval()
            s_matrix = np.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                use_quantile_sampling = self._use_quantile_sampling(train_or_test)
                if not use_quantile_sampling:
                    step = -1

                _, a = self.get_best_qvalue_and_action(s, step=step,
                                                       action_to_sample=action_to_sample)
            self.train()
            return a.numpy()

    def gaussian_policy(self, s, episode, step, train_or_test):
        """
        Given state s, at episode, take random action with p=eps if training Note - epsilon is determined by episode
        :param s:
        :param episode:
        :param step:
        :param train_or_test:
        :return:
        """
        assert train_or_test in ["train", "test"]
        # action_to_sample = self.action_to_sample if train_or_test == 'train' else self.interaction_action_to_sample
        action_to_sample = self.interaction_action_to_sample
        self.eval()
        s_matrix = np.array(s).reshape(1, self.state_size)
        use_quantile_sampling = self._use_quantile_sampling(train_or_test)
        if not use_quantile_sampling:
            step = -1

        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s, step=step, action_to_sample=action_to_sample)
            a = a.cpu().numpy()
        self.train()
        if train_or_test == "train":
            noise = np.random.normal(loc=0.,
                                     scale=self.params["noise_std"],
                                     size=len(a))
            a = np.clip(a + noise, -self.max_a, self.max_a)

        return a

    def update(self, target_Q, step=-1):

        # dictionary to store things like batch average Q and batch average Q star
        update_param = {"average_Q": 0, "average_Q_star": 0, "fourier_coefficients": None}
        if len(self.buffer_object) < self.params['batch_size']:
            return 0, update_param
        s_matrix, a_matrix, r_matrix, done_matrix, sp_matrix = self.buffer_object.sample(self.params['batch_size'])
        # clip rewards
        r_matrix = np.clip(r_matrix, a_min=-self.params['reward_clip'], a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        Q_star = target_Q.compute_Q_star(sp_matrix,
                                         quantile_sampling=self.params['use_quantile_sampling_bootstrapping'],
                                         step=step, do_minq=self.params['minq'])

        Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star

        if self.params.get("log_reward_function", None):
            update_param["fourier_coefficients"] = self.get_coefficients(self._coefficient_module_1,
                                                                         self.visualization_state)

        # prediction network
        y_hat = self.forward(s_matrix, a_matrix)

        update_param["average_Q_star"] = Q_star.mean().cpu().item()
        update_param["average_Q"] = y_hat.mean().cpu().item()

        loss = self.criterion(y_hat, y)

        if self.params['minq']:
            y_hat2 = self.forward(s_matrix, a_matrix, \
                                  coefficients=self.get_coefficients(self._coefficient_module_2, s_matrix))
            loss += self.criterion(y_hat2, y)
            update_param["average_Q"] = (y_hat + y_hat2).mean().cpu().item() / 2.

        if self.params["regularize"]:
            coeff_regularize_weights = self.inner_arg_freq.sum(dim=1) ** 2
            if self.params['functional'] == 'fourier':
                # there are 2x the # of coefficients for fourier to account for sin/cos coefficients.
                coeff_regularize_weights = coeff_regularize_weights.repeat((1, 2))
            outputs_to_regularize = self._coefficient_module_1(s_matrix) ** 2  # coefficients squared
            product_matrix = outputs_to_regularize.to(self.device).mul(coeff_regularize_weights.to(self.device))
            loss += self.params["regularization_weight"] * product_matrix.mean()

        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        utils_for_q_learning.sync_networks(
            target=target_Q,
            online=self,
            alpha=self.params['target_network_learning_rate'],
            copy=False)
        loss_data = loss.cpu().item()
        return loss_data, update_param


class FourierAgent(BaseAgent):
    """
    The Fourier basis agent
    """
    num_coefficient_scale = 2  # The scale factor for different basis

    def __init__(self, params, env, device, seed, centralized=False):
        super(FourierAgent, self).__init__(params, env, device, seed, centralized)

    def get_actions_and_basis_representations(self, actions=None, prefetch_basis_shape=None):
        """
        If we pass in actions, then we just get the representations for those actions. Then return the
        actions and basis reprsentations for those.
        If we pass in the shape, we select actions randomly from the list of pre-generated, and return
        those along with their selections. Easy peasy.

        Arguments:
            actions:
                If you want to calculate the basis-representations on-the-fly, you pass in actions.
            prefetch_basis_shape:
                If you want to use precomputed, you pass in the shape. Refers to the number of actions/
                representations you want
        """
        if actions is not None:
            # The basis representations of actions don't depend at all on the network. So,
            #  we should be able to compute them completely separately and use them over and over. 
            inner_arg_freq = torch.transpose(self.inner_arg_freq, dim0=0, dim1=1).to(
                self.device)  # [act_dim x num_basis]
            if len(actions.shape) == 3:
                inner_arg_freq = inner_arg_freq.unsqueeze(0)  # add another dimension for num_samples.
            actions_x_freqs = actions.matmul(inner_arg_freq.type(torch.float32))
            actions_x_freqs = actions_x_freqs * (np.pi / self.L)

            sines = torch.sin(actions_x_freqs)  # [batch_dim x action_to_sample x self.rank**self.act_dim]
            cosines = torch.cos(actions_x_freqs)  # [batch_dim x action_to_sample x self.rank**self.act_dim]

            sines_cat_cosines = torch.cat((sines, cosines),
                                          dim=-1)  # [batch_dim, action_to_sample x (2*self.rank**self.act_dim)]

            return actions, sines_cat_cosines
        else:
            num_to_fetch = np.prod(prefetch_basis_shape)
            reshape_shape = list(prefetch_basis_shape) + [-1]  # Because the final dimension will be the action_dim.
            indices = torch.randint(0, self._num_precomputed, size=(num_to_fetch,), device=self.device)
            actions = self.precomputed_actions[indices].reshape(reshape_shape)
            basis_representations = self.precomputed_basis_representations[indices].reshape(reshape_shape)
            return actions, basis_representations


class PolynomialAgent(BaseAgent):
    """
    The Polynomial basis agent
    """
    num_coefficient_scale = 1  # The scale factor for different basis

    def __init__(self, params, env, device, seed, centralized=False):
        super(PolynomialAgent, self).__init__(params, env, device, seed, centralized)

    def get_actions_and_basis_representations(self, actions=None, prefetch_basis_shape=None):
        if actions is not None:
            inner_arg_freq = self.inner_arg_freq.to(self.device)
            actions_repeated = torch.repeat_interleave(actions.unsqueeze(-1), inner_arg_freq.shape[0], dim=-1)
            actions_power = torch.pow(actions_repeated, inner_arg_freq.T)
            actions_prod = torch.prod(actions_power, dim=-2)
            return actions, actions_prod
        else:
            num_to_fetch = np.prod(prefetch_basis_shape)
            reshape_shape = list(prefetch_basis_shape) + [-1]  # Because the final dimension will be the action_dim.
            indices = torch.randint(0, self._num_precomputed, size=(num_to_fetch,), device=self.device)
            actions = self.precomputed_actions[indices].reshape(reshape_shape)
            basis_representations = self.precomputed_basis_representations[indices].reshape(reshape_shape)
            return actions, basis_representations


class LegendreAgent(BaseAgent):
    """
    The Legendre basis agent
    """
    num_coefficient_scale = 1  # The scale factor for different basis

    def __init__(self, params, env, device, seed, centralized=False):
        self.rank = params['rank']
        self.device = device
        self.legendre_coefficients = self.compute_legendre_matrix(self.rank).to(self.device)
        self.powers_needed = torch.from_numpy(np.array(list(range(0, self.rank + 1)))).to(self.device)
        super(LegendreAgent, self).__init__(params, env, device, seed, centralized)

    def compute_legendre_matrix(self, rank):
        max_order = rank + 1
        coefficients_for_orders = []
        for order in range(max_order):
            legendre_poly = np.polynomial.legendre.Legendre.basis(order, [-1, 1]).convert(kind=Polynomial)
            legendre_coefs = list(legendre_poly.coef)
            coefficients = legendre_coefs + [0] * (max_order - len(legendre_coefs))
            coefficients_for_orders.append(np.array(coefficients))

        legendre_coefficients = np.array(coefficients_for_orders)
        legendre_coefficients = torch.from_numpy(legendre_coefficients).to(torch.float32).to(self.device)
        legendre_coefficients = legendre_coefficients.reshape(1, 1, *legendre_coefficients.shape)
        return legendre_coefficients

    def get_actions_and_basis_representations(self, actions=None, prefetch_basis_shape=None):
        if actions is not None:
            shape = actions.shape
            if len(shape) == 2:
                actions = actions.unsqueeze(dim=0)  # [1 x 500 x 1]
            inner_arg_freq = self.inner_arg_freq.to(self.device)

            actions_repeated = torch.repeat_interleave(actions.unsqueeze(-1), self.powers_needed.shape[0], dim=-1)
            actions_raised = torch.pow(actions_repeated, self.powers_needed)
            actions_raised = torch.transpose(actions_raised, -1, -2).to(torch.float32)

            evaluated_1_d_legendre_lookup = torch.matmul(self.legendre_coefficients, actions_raised)
            evaluated_1_d_legendre_lookup = torch.transpose(evaluated_1_d_legendre_lookup, -1, -2)

            unsqueezed_inner_arg_frequency = inner_arg_freq.T.reshape(1, 1, *inner_arg_freq.T.shape).to(torch.int64)
            indices = torch.repeat_interleave(unsqueezed_inner_arg_frequency,
                                              repeats=evaluated_1_d_legendre_lookup.shape[1], dim=1)
            indices = torch.repeat_interleave(indices, repeats=evaluated_1_d_legendre_lookup.shape[0], dim=0)
            accumulator = torch.transpose(torch.gather(evaluated_1_d_legendre_lookup, dim=-1, index=indices), -1, -2)
            legendre_values_before_net_mul = torch.prod(accumulator, dim=-1).to(torch.float32)
            if len(shape) == 2:
                actions = actions.squeeze()
                legendre_values_before_net_mul = legendre_values_before_net_mul.squeeze()
            return actions, legendre_values_before_net_mul
        else:
            num_to_fetch = np.prod(prefetch_basis_shape)
            reshape_shape = list(prefetch_basis_shape) + [-1]  # Because the final dimension will be the action_dim.
            indices = torch.randint(0, self._num_precomputed, size=(num_to_fetch,), device=self.device)
            actions = self.precomputed_actions[indices].reshape(reshape_shape)
            basis_representations = self.precomputed_basis_representations[indices].reshape(reshape_shape)
            return actions, basis_representations
