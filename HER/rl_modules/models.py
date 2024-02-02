import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from alphazero.network.distributions import SquashedNormal, GeneralizedBeta
from alphazero.network.kl_divergences import (
    DiagonalNormal_kl_divergence, 
    DiagonalNormal_rkl_divergence, 
    DiagonalNormal_square_divergence,
    DiagonalNormal_js_divergence_log_std,
)
from alphazero.network.utils import (
    _map_nonlinearities,
    _process_str,
)
from torch.distributions.normal import Normal

from alphazero.network.distributions import SquashedNormal, GeneralizedBeta
import numpy as np
import ipdb


LOG_STD_MAX = 2
LOG_STD_MIN = -20

clip_max = 3


# def weights_init_(m):
#     return
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)


class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, env_params['action'])
        self.log_std_layer = nn.Linear(256, env_params['action'])
        
        self.mu_layer.weight.data.fill_(0)
        self.mu_layer.bias.data.fill_(0)
        self.log_std_layer.weight.data.fill_(0)
        self.log_std_layer.bias.data.fill_(0.)

    def forward(self, x, with_logprob = False, deterministic = False, forced_exploration=1, 
            full_distribution=False):
        # with_logprob = False
        mu, log_std = self.get_distribution(x)
        rv = self.sample_distribution(mu, log_std, with_logprob = with_logprob, 
            deterministic = deterministic, forced_exploration=forced_exploration)
        if full_distribution: 
            return rv + (mu, std)
        else: 
            return rv
        # return actions

    def get_distribution(self, x):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -20
        # clip_max = 3
        clip_max = 50
        x = torch.clamp(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        net_out = F.relu(self.fc3(x))


        mu = self.mu_layer(net_out)#/100
        log_std = self.log_std_layer(net_out)#-1#/100 -1.
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample_distribution(self, mu, log_std,
            with_logprob = False, deterministic = False, forced_exploration=1):
        
        std = torch.exp(log_std)*forced_exploration
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(torch.log(torch.tensor(2.)) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = pi_action.clamp(-10, 10)
        pi_action = torch.tanh(pi_action)
        pi_action = self.max_action * pi_action

        assert (pi_action <= self.max_action).all()
        assert (pi_action >= -self.max_action).all()

        # import ipdb
        # ipdb.set_trace()
        if with_logprob: 
            return pi_action, logp_pi
        else: 
            return pi_action

            
    def set_normalizers(self, o, g): 
        self.o_norm = o
        self.g_norm = g

    def _get_norms(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        return obs_norm, g_norm

    def _get_denorms(self, obs, g):
        obs_denorm = self.o_norm.denormalize(obs)
        g_denorm = self.g_norm.denormalize(g)
        return obs_denorm, g_denorm

    def normed_forward(self, obs, g, deterministic=False): 
        obs_norm, g_norm = self._get_norms(torch.tensor(obs, dtype=torch.float32), torch.tensor(g, dtype=torch.float32))
        # concatenate the stuffs
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)
        return self.forward(inputs, deterministic=deterministic, forced_exploration=1)



class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.max_action = env_params['action_max']
        self.norm1 = nn.LayerNorm(env_params['obs'] + 2*env_params['goal'] + env_params['action'])
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        # pdb.set_trace()
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = self.norm1(x)
        x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class T_conditioned_ratio_critic(nn.Module):
    def __init__(self, env_params):
        super(T_conditioned_ratio_critic, self).__init__()
        self.env_params = env_params
        self.max_action = env_params['action_max']
        input_shape = env_params['obs'] + 2*env_params['goal'] + 1 + env_params['action']
        self.norm1 = nn.LayerNorm(input_shape)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(input_shape, 256)
        # self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        # self.her_goal_range = (env_params['obs'] + env_params['goal'] -1, env_params['obs'] + 2*env_params['goal']-1)
        self.her_goal_range = (env_params['obs'] , env_params['obs'] + env_params['goal'])
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.p_out = nn.Linear(256, 1)

    def forward(self, x, T, actions, return_p=False):
        mult_val = torch.ones_like(x)
        new_x = torch.cat([x*mult_val, T, actions / self.max_action], dim=1)
        assert new_x.shape[0] == x.shape[0] and new_x.shape[-1] == (x.shape[-1] + 1 + self.env_params['action'])
        x = new_x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.norm2(x)
        x = F.relu(self.fc2(x))
        x = self.norm3(x)
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x) 
        if return_p: 
            #exponentiate p to ensure it's non-negative
            exp = .5 #exponent for p
            base = 4 #Initially give states small probability. 
                # If they're not visited, they won't be updated, so they should remain small
                # States that are visited will grow, which is what we want
            p_value =  2**(exp*self.p_out(x) - base)
            # p_value =  self.p_out(x) #+ 1
            return q_value, p_value
        else: 
            return q_value


class test_T_conditioned_ratio_critic(nn.Module):
    def __init__(self, env_params):
        super(test_T_conditioned_ratio_critic, self).__init__()
        self.max_action = env_params['action_max']
        # self.norm1 = nn.LayerNorm(env_params['obs'] + 2*env_params['goal'] + 1 + env_params['action'])
        self.norm = False
        if self.norm:
            self.norm1 = nn.LayerNorm(256)
            self.norm2 = nn.LayerNorm(256)
            self.norm3 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(env_params['obs'] + 2*env_params['goal'] + 1 + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        self.p_out = nn.Linear(256, 1)

    # def forward(self, x, actions,  T=0, return_p=False):
    #     T = torch.zeros(x.shape[:-1] + (1,))
    def forward(self, x, T, actions, return_p=False):
        # pdb.set_trace()
        t_scale = .01
        x = torch.cat([x, T*t_scale, actions / self.max_action], dim=1)
        # x = torch.clip(x, -clip_max, clip_max)
        x = F.relu(self.fc1(x))
        if self.norm: x = self.norm1(x)
        x = F.relu(self.fc2(x))
        if self.norm: x = self.norm2(x)
        x = F.relu(self.fc3(x))
        if self.norm: x = self.norm3(x)
        q_value = self.q_out(x) 
        if return_p: 
            #exponentiate p to ensure it's non-negative
            exp = .5 #exponent for p
            base = 4 #Initially give states small probability. 
                # If they're not visited, they won't be updated, so they should remain small
                # States that are visited will grow, which is what we want
            val =  (exp*self.p_out(x) - base)
            p_value = torch.nn.ELU()(val) + 1
            # p_value =  self.p_out(x) #+ 1
            return q_value, p_value
        else: 
            return q_value

        # return q_value






def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])



class StateValueEstimator(nn.Module):
    def __init__(self, actor, critic, gamma):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.gamma = gamma

    def q2time(self, q):
        # max_q = 1/(1-self.args.gamma)
        # ratio = -.99*torch.clip(q/max_q, -1,0) #.99 for numerical stability
        return torch.log(1+q*(1-self.gamma)*.998)/torch.log(torch.tensor(self.gamma))

    def forward(self, o: torch.Tensor, g: torch.Tensor, norm=True): 
        assert type(o) == torch.Tensor
        assert type(g) == torch.Tensor
        if norm: 
            obs_norm, g_norm = self.actor._get_norms(o,g)
        else: 
            obs_norm, g_norm = o, g
        inputs = torch.cat([obs_norm, g_norm])
        inputs = inputs.unsqueeze(0)

        action = self.actor(inputs)
        value = self.critic(inputs, action).squeeze()

        # return self.q2time(value)
        return value



# class PolicyInterface(StateValueEstimator):
#     def __init__(self, actor, critic, gamma, env_params=None):
#         super().__init__(actor, critic, gamma)
#         self.action_dim=env_params['action']
#         self.state_dim =env_params['obs'] + env_params['goal']

#     @torch.no_grad()
#     def sample_action(self, x: torch.Tensor, log_prob=False, deterministic=False):
#     #(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
#         # with_logprob = False
#         mu, log_std = self.get_distribution(x)
#         action, lp  = self.actor.sample_distribution(mu, log_std, with_logprob = True, 
#             deterministic = deterministic)
#         if log_prob:
#             return action.detach().cpu().numpy(), lp.detach().cpu().numpy()
#         else:
#             return action.detach().cpu().numpy()

#     def density(self, x, y):
#         return torch.tensor(1)

#     def normalize_input_vec(self, x):
#         assert self.actor.o_norm.mean.shape[-1] + self.actor.g_norm.mean.shape[-1] == x.shape[-1]
#         o, g = x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)
#         obs_norm, g_norm = self.actor._get_norms(o,g)
#         try: 
#             inputs = torch.cat([obs_norm, g_norm], dim=-1)
#         except: 
#             import ipdb
#             ipdb.set_trace()
#         # inputs = inputs.unsqueeze(0)
#         return inputs

#     def normalize_input_vec_for_critic(self, x):
#         assert self.actor.o_norm.mean.shape[-1] + self.actor.g_norm.mean.shape[-1] == x.shape[-1]
#         o, g = x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)
#         norm = True
#         if norm: 
#             obs_norm, g_norm = self.actor._get_norms(o,g)
#         else: 
#             obs_norm, g_norm = o, g

#         inputs = torch.cat([obs_norm, g_norm, g_norm], dim=-1)
#         return inputs

#     def get_distribution(self, x):
#         normed_x = self.normalize_input_vec(x)
#         # normed_x = norm_whole_vec(x)
#         assert x.shape == normed_x.shape
#         mu, log_std = self.actor.get_distribution(normed_x)
#         return mu, log_std

#     def forward(self, x: torch.FloatTensor):
#         if len(x.shape) == 1:
#             original_x = x
#             x = x.unsqueeze(0)
#             resqueeze = True
#         else: 
#             resqueeze = False

#         mu, log_std = self.get_distribution(x)
#         sigma = log_std.exp()
#         state = self.normalize_input_vec_for_critic(x)
#         T = torch.tensor([0]*state.shape[0]).float()
#         T = T.unsqueeze(-1)
#         value = self.critic(state, T, mu).squeeze()

#         # import ipdb
#         # if len(mu.shape) != 2:
#         #     ipdb.set_trace()
#         if resqueeze: 
#             mu = mu.squeeze()
#             sigma = sigma.squeeze()

        
#         return mu, sigma, value

#     def predict_V(self, x):
#         _, _, V = self(x)
#         return V.detach()

#     ##################################################################################################

#     def _preproc_inputs(self, obs, g, gpi=None):
#         obs_norm, g_norm = self.actor._get_norms(obs,g)
#         # concatenate the stuffs
#         if gpi is not None: 
#             gpi_norm = self.g_norm.normalize(g)
#             inputs = torch.cat([obs_norm, g_norm, gpi_norm])
#         else:
#             inputs = torch.cat([obs_norm, g_norm])
#         inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
#         return inputs

#     def sample_og(self, o, g):
#         preprocced_inputs = self._preproc_inputs(o,g)
#         pi = self.actor(preprocced_inputs)
#         actions_tensor = pi.detach().cpu()

#         return actions_tensor

#     def split_vec(self, x):
#         return x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)

#     def norm_whole_vec(self, x):
#         obs, g = self.split_vec(x)
#         return self._preproc_inputs(obs, g)

class PolicyInterface(StateValueEstimator):
    def __init__(self, actor, critic, gamma, env_params=None):
        super().__init__(actor, critic, gamma)
        self.action_dim=env_params['action']
        self.state_dim =env_params['obs'] + env_params['goal']
        
        # layers = [
        #     nn.Linear(self.state_dim*2, hidden_dimensions[0]),
        #     activation(inplace=True),
        # ]
        # if layernorm:
        #     layers.append(nn.LayerNorm(normalized_shape=hidden_dimensions[0]))

        # if 1 < self.hidden_layers:
        #     for i, hidden_dim in enumerate(hidden_dimensions[:-1]):
        #         hid = [
        #             nn.Linear(hidden_dim, hidden_dimensions[i + 1]),
        #             activation(inplace=True),
        #         ]
        #         if layernorm:
        #             hid.append(nn.LayerNorm(normalized_shape=hidden_dimensions[i + 1]))
        #         layers.extend(hid)
        #     final_layer = nn.Linear(hidden_dimensions[-1], 1)

        # self.density_trunk = nn.Sequential(*layers)
        # self.density_head = nn.Sequential(*[
        #         final_layer, 
        #         nn.Softplus()
        #     ])

        # self.inv_density_head = nn.Sequential(*[
        #         final_layer, 
        #         nn.Softplus()
        #     ])
        self.density_head = nn.Linear(self.state_dim*2, 1)
        # self.density_head = nn.Linear(self.state_dim, 1)
        self.density_head.weight.data.fill_(0)
        self.density_head.bias.data.fill_(1)



    def density(self, start_state, end_state, inv_density = False) -> torch.FloatTensor:

        upper = start_state
        upper = end_state
        lower = end_state
        input_tensor = torch.cat((upper, lower), dim=-1)
        dens = self.density_head(input_tensor.float())
        # dens = torch.sum(input_tensor, dim=-1)*0 + 1
        dens = dens*0 + 1
        if inv_density:
            return dens, 1/dens
        else: 
            return dens

    @torch.no_grad()
    def sample_action(self, x: torch.Tensor, log_prob=False, deterministic=False):
    #(self, x, with_logprob = False, deterministic = False, forced_exploration=1):
        # with_logprob = False
        mu, log_std = self.get_distribution(x)
        action, lp  = self.actor.sample_distribution(mu, log_std, with_logprob = True, 
            deterministic = deterministic)
        if log_prob:
            return action.detach().cpu().numpy(), lp.detach().cpu().numpy()
        else:
            return action.detach().cpu().numpy()

    # def density(self, x, y):
    #     return torch.tensor(1)

    def normalize_input_vec(self, x):
        assert self.actor.o_norm.mean.shape[-1] + self.actor.g_norm.mean.shape[-1] == x.shape[-1]
        o, g = x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)
        obs_norm, g_norm = self.actor._get_norms(o,g)
        try: 
            inputs = torch.cat([obs_norm, g_norm], dim=-1)
        except: 
            import ipdb
            ipdb.set_trace()
        # inputs = inputs.unsqueeze(0)
        return inputs

    def normalize_input_vec_for_critic(self, x):
        assert self.actor.o_norm.mean.shape[-1] + self.actor.g_norm.mean.shape[-1] == x.shape[-1]
        o, g = x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)
        norm = True
        if norm: 
            obs_norm, g_norm = self.actor._get_norms(o,g)
        else: 
            obs_norm, g_norm = o, g

        inputs = torch.cat([obs_norm, g_norm, g_norm], dim=-1)
        return inputs

    def get_distribution(self, x):
        normed_x = self.normalize_input_vec(x)
        # normed_x = norm_whole_vec(x)
        assert x.shape == normed_x.shape
        mu, log_std = self.actor.get_distribution(normed_x)
        return mu, log_std

    def forward(self, x: torch.FloatTensor):
        mu, log_std, value = self.forward_log_sigma(x)
        sigma = log_std.exp()
        return mu, sigma, value

    def forward_log_sigma(self, x):
        if len(x.shape) == 1:
            original_x = x
            x = x.unsqueeze(0)
            resqueeze = True
        else: 
            resqueeze = False

        mu, log_std = self.get_distribution(x)
        state = self.normalize_input_vec_for_critic(x)
        T = torch.tensor([0]*state.shape[0]).float()
        T = T.unsqueeze(-1)
        value = self.critic(state, T, mu).squeeze()

        # import ipdb
        # if len(mu.shape) != 2:
        #     ipdb.set_trace()
        if resqueeze: 
            mu = mu.squeeze()
            log_std = log_std.squeeze()

        return mu, log_std, value

    def predict_V(self, x):
        _, _, V = self(x)
        return V.detach()

    ##################################################################################################

    def _preproc_inputs(self, obs, g, gpi=None):
        obs_norm, g_norm = self.actor._get_norms(obs,g)
        # concatenate the stuffs
        if gpi is not None: 
            gpi_norm = self.g_norm.normalize(g)
            inputs = torch.cat([obs_norm, g_norm, gpi_norm])
        else:
            inputs = torch.cat([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        return inputs

    def sample_og(self, o, g):
        preprocced_inputs = self._preproc_inputs(o,g)
        pi = self.actor(preprocced_inputs)
        actions_tensor = pi.detach().cpu()

        return actions_tensor

    def split_vec(self, x):
        return x.split([self.actor.o_norm.mean.shape[-1], self.actor.g_norm.mean.shape[-1]], dim=-1)

    def norm_whole_vec(self, x):
        obs, g = self.split_vec(x)
        return self._preproc_inputs(obs, g)


    def get_train_data(self, states, actions):
        mu, sigma, V_hat = self(states)
        # This aligns the distribution batch_shape with the number of actions at the root
        # It can be thought of as generating num_actions identical normal distributions for each agent
        # and then sampling the log_prob for action from the distribution
        # num_actions = actions.shape[-1]
        # mu = mu.expand((-1, num_actions))
        # sigma = sigma.expand((-1, num_actions))
        if len(actions.shape) > len(mu.shape):
            reshape = True
        else: 
            reshape = False
        # reshape = False
            # reshape = True
            # actions = actions.unsqueeze(-1)
        if reshape:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)

        self.action_bound = 1.
        if self.action_bound:
            normal = SquashedNormal(mu, sigma, self.action_bound)
        else:
            normal = D.Normal(mu, sigma)

        log_probs = normal.log_prob(actions)
        if reshape:
            log_probs = log_probs.sum(-1)

        entropy = -log_probs.mean(dim=-1)

        return log_probs, entropy, V_hat.unsqueeze(-1)


    def get_train_data_generalized(self, input_dict):
        self.log_param_max = 2
        self.log_param_min = -20
        states = input_dict['states']
        actions = input_dict['actions']
        # mu, sigma, V_hat = self(states)
        mu, log_std, V_hat = self.forward_log_sigma(states)
        log_sigma = torch.clamp(log_std, min=self.log_param_min, max=self.log_param_max)
        sigma = log_std.exp()
        # This aligns the distribution batch_shape with the number of actions at the root
        # It can be thought of as generating num_actions identical normal distributions for each agent
        # and then sampling the log_prob for action from the distribution
        # num_actions = actions.shape[-1]
        # mu = mu.expand((-1, num_actions))
        # sigma = sigma.expand((-1, num_actions))
        self.action_bound = 1.
        if self.action_bound:
            normal = SquashedNormal(mu, sigma, self.action_bound)
        else:
            normal = D.Normal(mu, sigma)

        # log_probs = normal.log_prob(torch.stack(actions, dim=0))
            # log_probs = [normal.log_prob(actions[i])[i] for i in range(len(actions))]
        log_probs = []
        for i in range(len(actions)):
            dist = D.Normal(mu[i], sigma[i])
            try: 
                log_prob = dist.log_prob(actions[i])
                if len(log_prob.shape) > 1:
                    log_prob = log_prob.sum(dim=-1)
                log_probs.append(log_prob)
            except: 
                import ipdb
                ipdb.set_trace()
        # log_probs = 
        mean = lambda x: 0 if len(x) == 0 else sum(x)/len(x)
        # entropy = -log_probs.mean(dim=-1)
        entropy = torch.stack([-lp.mean() for lp in log_probs], dim=0)#log_probs.mean(dim=-1)

        initial_mu = input_dict['mu']
        initial_sigma = input_dict['sigma']

        output_dict = {}
        output_dict['target_policy'] = None#??
        output_dict['policy_loss'] = None#??
        output_dict['log_probs'] = log_probs#??
        output_dict['probs'] = [torch.exp(lp) for lp in log_probs]
        # output_dict['probs'] = [lp*0 + 1 for lp in log_probs]

        output_dict['kl_divergence'] = DiagonalNormal_kl_divergence(
            (initial_mu, initial_sigma), (mu, sigma))
        output_dict['base_kl_divergence'] = DiagonalNormal_kl_divergence(
            (mu, sigma), (initial_mu*0, initial_sigma*0+1))
            #RKL from baseline distribution
            #Here the baseline distribution is a unit gaussian
        output_dict['square_divergence'] = DiagonalNormal_square_divergence(
            (mu, log_std), (initial_mu*0, initial_sigma*0))
        output_dict['js_divergence'] = DiagonalNormal_js_divergence_log_std(
            (initial_mu, initial_sigma), (mu, log_std))
        output_dict['base_js_divergence'] = DiagonalNormal_js_divergence_log_std(
            (initial_mu*0, initial_sigma*0), (mu, log_std))

        output_dict['entropy'] = entropy#??
        output_dict['V_hat'] = V_hat#??

        if 'lo' in input_dict.keys() and 'hi' in input_dict.keys():
            # output_dict['density'] = self.density(input_dict['lo'], input_dict['hi'])
            assert input_dict['lo'].shape == input_dict['states'].shape
            # output_dict['density'] = self.density(input_dict['states'])

            # output_dict['density'] = self.density(input_dict['root_state'], input_dict['states'])
            # interp = torch.tensor(np.random.rand(input_dict['lo'].shape[-1]))
            # random_state = interp*input_dict['lo'] + (1-interp)*input_dict['hi']
            interp = torch.tensor(np.random.uniform(size=input_dict['lo'].shape))
            random_state = interp
            interp = torch.tensor(np.random.uniform(size=input_dict['lo'].shape))
            random_kd_region_state = interp*input_dict['lo'] + (1-interp)*input_dict['hi']
            output_dict['density'] = self.density(input_dict['root_state'], random_state)
            output_dict['node_density'], output_dict['inv_density'] = self.density(input_dict['root_state'], input_dict['states'], inv_density=True)
            # except: 
            #     import ipdb
            #     ipdb.set_trace()
            # output_dict['node_density'], output_dict['inv_density'] = self.density(input_dict['root_state'], input_dict['states'], inv_density=True)
            # output_dict['node_density'], output_dict['inv_density'] = self.density(input_dict['root_state'], random_kd_region_state, inv_density=True)

        return output_dict


class value(nn.Module):
    def __init__(self, env_params):
        from torch.nn.utils.parametrizations import spectral_norm
        super(value, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.fc2 = spectral_norm(nn.Linear(256, 256))
        self.fc3 = spectral_norm(nn.Linear(256, 256))
        self.q_out = nn.Linear(256, 1)

        # self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value


class PolicyInterfaceV(PolicyInterface):
    def forward_log_sigma(self, x):
        if len(x.shape) == 1:
            original_x = x
            x = x.unsqueeze(0)
            resqueeze = True
        else: 
            resqueeze = False

        mu, log_std = self.get_distribution(x)
        state = self.normalize_input_vec(x)
        value = self.critic(state).squeeze()

        # import ipdb
        # if len(mu.shape) != 2:
        #     ipdb.set_trace()
        if resqueeze: 
            mu = mu.squeeze()
            log_std = log_std.squeeze()

        return mu, log_std, value