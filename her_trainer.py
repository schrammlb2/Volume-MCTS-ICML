from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import trange
# import wandb
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.agent.losses import A0CLossTuned, A0CLoss
from alphazero.helpers import check_space, store_actions
from rl.make_game import make_game
import global_config
from utils.logger import Logger

import gymnasium
from gymnasium.wrappers import StepAPICompatibility
from rl.wrappers import ResetCompatibilityWrapper

wifi = False
# wifi = True
name="continuous_external_training_HER"

from get_az_args import get_az_args

from alphazero.agent.agents import *
from alphazero.search.mcts import *

from trainer import Trainer

class HERTrainer(Trainer):
    def __init__(self, cfg, Env, unwrapped_env):
        self.cfg = cfg
        self.buffer = hydra.utils.instantiate(cfg.buffer)
        self.buffer.clear()
        self.rollouts = cfg.mcts.n_rollouts


        cfg.mcts.n_rollouts = 2
        self.agent = hydra.utils.instantiate(cfg.agent)


        self.args = cfg.her
        self.args.n_batches = 100
        self.args.n_epochs = 1
        self.her_agent = self.make_her_agent(self.args, unwrapped_env)

        # her_agent.learn()

        self.agent.set_network(self.her_agent.get_policy_interface())


    def get_env_params(self, env):
        obs = env.reset()
        if type(obs) == tuple:
            obs = obs[0]
        # close the environment
        params = {'obs': obs['observation'].shape[0],
                'goal': obs['desired_goal'].shape[0],
                'action': env.action_space.shape[0],
                'action_max': env.action_space.high[0],
                }
        try:
            params['max_timesteps'] = self.cfg.env.max_episode_length
        except: 
            params['max_timesteps'] = 50
            e = env
            while hasattr(e, "env") and (type(e) != gymnasium.wrappers.time_limit.TimeLimit):
                e = e.env            
            if type(e) == gymnasium.wrappers.time_limit.TimeLimit:
                params['max_timesteps'] = e._max_episode_steps
        return params

    def make_her_agent(self, args, env):
        from HER.arguments import get_args 
        from HER.rl_modules.models import PolicyInterface
        if self.cfg.standard_her: 
            from HER.rl_modules.usher_agent_high_dim import ddpg_agent 
        else: 
            from HER.rl_modules.usher_nonepisodic_her import ddpg_agent 

        env_params = self.get_env_params(env)
        ddpg_trainer = ddpg_agent(args, env, env_params, write=False)
        return ddpg_trainer


    def test_model(self, her_agent):
        her_agent.actor_network.eval()
        ev = her_agent._eval_agent()
        her_agent.actor_network.train()
        success_rate, reward, value = ev['success_rate'], ev['reward_rate'], ev['value_rate']
        
        print(f'[{datetime.now()}] epoch is: {ep}, '
            f'eval success rate is: {success_rate:.3f}, '
            f'average reward is: {reward:.3f}, '
            f'average value is: {value:.3f}')

    def reset(self):
        raise NotImplemented        

    def _run(self, Env, epoch):
        if global_config.using_gymnasium:
            state = Env.reset()[0]
        else:
            state = Env.reset()
        R = 0.0  # Total return counter
        self.agent.reset_mcts(root_state=state)

        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
        ep_col = []

        episode_dict = {
            'obs': [],
            'obs_next': [],
            'actions': [],
            'ag': [],
            'ag_next': [],
            'g': [],
            'her_goals': [],
            't_remaining': [],
            'col': [],
        }
        # reset the environment
        observation = Env.get_goal_obs()
        obs = observation['observation'].copy()
        ag = observation['achieved_goal'].copy()
        g = observation['desired_goal'].copy()

        self.agent.reset_mcts(root_state=state)
        self.agent.epoch = epoch
        
        success = False
        final_R = 0
        for t in range(self.cfg.max_episode_length):
            # MCTS step
            # run mcts and extract the root output
            action, s, actions, counts, Qs, V = self.agent.act(Env=Env)
            if len(actions.shape) == 1:
                actions.reshape(-1, 1)
            # import ipdb
            # ipdb.set_trace()
            self.buffer.store((s, actions, counts, Qs, V))

            # Make the true step
            action = action.reshape(-1)
            state, step_reward, terminal, _ = Env.step(action)

            observation_new = Env.get_goal_obs()
            obs_new = observation_new['observation'].copy()
            ag_new = observation_new['achieved_goal'].copy()
            # append rollouts
            ep_obs.append(obs.copy())
            ep_ag.append(ag.copy())
            ep_g.append(g.copy())
            ep_actions.append(action.copy())
            ep_col.append(False)

            obs = obs_new
            ag = ag_new

            R += step_reward
            
            # if terminal and final_R == 0:
            #     terminal_reward = step_reward*(self.cfg.max_episode_length - 1 - t)
            #     R += step_reward
            #     final_R = terminal_reward
            #     success = True
            #     # break
            # else:
            #     # reset the mcts as we can't reuse the tree
            #     self.agent.reset_mcts(root_state=state)


            if step_reward > 0:
                final_R = max(final_R, step_reward*(self.cfg.max_episode_length - 1 - t))
                
            self.agent.reset_mcts(root_state=state)
            # if t >= self.cfg.max_episode_length - 1:
            #     if self.cfg.standard_her and self.cfg.train_her:
            #         pass
            #     else: 
            #         R += step_reward*(self.cfg.max_episode_length - 1 - t)
            #         break
            # else:
            #     # reset the mcts as we can't reuse the tree
            #     self.agent.reset_mcts(root_state=state)

        success = (final_R > 0)
        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())

        max_len = len(ep_obs)-1
        episode_list_of_dicts = [
            {
                'obs': ep_obs[i],
                'obs_next': ep_obs[i+1],
                'actions': ep_actions[i],
                'ag': ep_ag[i],
                'ag_next': ep_ag[i+1],
                'g': ep_g[i],
                'her_goals': [
                    ep_ag[np.random.randint(i,max_len+1)] 
                    for _ in range(int(self.args.replay_k))
                ],
                't_remaining': [max_len-i],
                'col': ep_col[i]
            }
            # for i in range(t-1)
            for i in range(max_len)
        ]
        if len(episode_list_of_dicts) > 0:
            for key in episode_list_of_dicts[0]:
                episode_dict[key] += [
                    dictionary[key]
                    for dictionary in episode_list_of_dicts
                ]

            for key in episode_list_of_dicts[0]:
                episode_dict[key] = np.array(episode_dict[key])

        self.her_dict = episode_dict
                    
        self.episode_data = [np.array([ep_obs]), np.array([ep_ag]), np.array([ep_g]), np.array([ep_actions])]

        return success, final_R

    # def train(self):    
    #     self.her_agent.learn()
    def train(self, n=1):    
        for _ in range(n):
            self.run(self.Env, 1)

    def run(self, Env, epoch):
        if callable(Env.unwrapped): unwrapped_env = Env.unwrapped()
        else: unwrapped_env = Env.unwrapped
        self.her_agent = self.make_her_agent(self.args, unwrapped_env)
        self.agent.set_network(self.her_agent.get_policy_interface())
        success, R = self._run(Env, epoch)
        for _ in range(self.rollouts):
            self.her_agent.perform_update(self.episode_data)
            success, R = self._run(Env, epoch)
        return success, R


@hydra.main(config_path="config", config_name=f"run_{name}")
def run_continuous_agent(cfg):

    print(f"Running {name}")
    # print(cfg)
    # return
    train_her = f"_her:{cfg.train_her}"
    train_az = f"_az:{cfg.train_az}"
    standard_her = f"_full_traj_her:{cfg.standard_her}"
    print(train_her)
    print(train_az)
    print(standard_her)

    log=False
    if log: 
        logger = Logger(["epoch", "R"], filename=name + train_her + train_az + standard_her)

    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []
    # Environments
    Env = make_game(cfg.env.game)
    Env.reset()

    # set seeds
    np.random.seed(cfg.seed)

    t_total = 0  # total steps

    # get environment info
    state_dim, _ = check_space(Env.observation_space)
    action_dim, action_discrete = check_space(Env.action_space)

    assert (
        action_discrete == False
    ), "Using continuous agent for a discrete action space!"

    # set config environment values
    cfg.policy.representation_dim = state_dim[0]
    cfg.policy.action_dim = action_dim[0]
    # assumes that all dimensions of the action space are equally bound
    cfg.policy.action_bound = float(Env.action_space.high[0])
    # cfg.policy.action_bound = None

    # cfg.game = cfg.env.game
    cfg.max_episode_length = cfg.env.max_episode_length
    cfg.mcts.n_rollouts = cfg.env.n_rollouts_per_step
    cfg.num_train_epochs = cfg.env.num_train_epochs

    cfg.num_train_episodes = cfg.env.num_train_episodes


    if cfg.policy.distribution == "beta":
        distribution = "Beta"
    elif cfg.policy.distribution == "normal" and cfg.policy.action_bound:
        distribution = "Squashed Normal"
    else:
        distribution = "Normal"

    config = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # "Environment": Env.unwrapped.spec.id,
        "Environment seed": cfg.seed,
        "Training episodes": cfg.num_train_episodes,
        "Episode length": cfg.max_episode_length,
        "Training epochs": cfg.num_train_epochs,
        "Batch size": cfg.buffer.batch_size,
        "Replay buffer size": cfg.buffer.max_size,
        "MCTS rollouts": cfg.mcts.n_rollouts,
        "UCT constant": cfg.mcts.c_uct,
        "Discount factor": cfg.mcts.gamma,
        "MCTS epsilon greedy": cfg.mcts.epsilon,
        "Progressive widening factor [c_pw]": cfg.mcts.c_pw,
        "Progressive widening exponent [kappa]": cfg.mcts.kappa,
        "V target policy": cfg.mcts.V_target_policy,
        "Final selection policy": cfg.agent.final_selection,
        "Agent epsilon greedy": cfg.agent.epsilon,
        "Network hidden layers": cfg.policy.hidden_dimensions,
        "Network hidden units": len(cfg.policy.hidden_dimensions),
        "Network nonlinearity": cfg.policy.nonlinearity,
        "LayerNorm": cfg.policy.layernorm,
        "Clamp log param": True,
        "Clamp loss": "Loss scaling",
        "Log prob scale": "Corrected entropy",
        "Num mixture components": cfg.policy.num_components,
        "Distribution": distribution,
        "Optimizer": "Adam"
        if cfg.optimizer._target_ == "torch.optim.Adam"
        else "RMSProp",
        "Learning rate": cfg.optimizer.lr,
        "Log counts scaling factor [tau]": cfg.agent.loss_cfg.tau,
        "Policy coefficient": cfg.agent.loss_cfg.policy_coeff,
        "Value coefficient": cfg.agent.loss_cfg.value_coeff,
        "Loss reduction": cfg.agent.loss_cfg.reduction,
    }



    config.update(
        {
            "Target entropy": -cfg.agent.loss_cfg.action_dim,
            "Loss lr": 0.001,
            "Loss type": "A0C loss tuned",
        }
    )

    if wifi: 
        run = wandb.init(name="A0C", project="a0c", config=config)
    else: 
        run = None

    # pbar = trange(cfg.num_train_episodes)
    n_runs = 10
    for _ in range(n_runs):
        if callable(Env.unwrapped):
            unwrapped_env = Env.unwrapped()
        else:
            unwrapped_env = Env.unwrapped

        unwrapped_env = StepAPICompatibility(unwrapped_env, output_truncation_bool=False)
        unwrapped_env = ResetCompatibilityWrapper(unwrapped_env)

        trainer = ContinuousTrainer(cfg, Env, unwrapped_env)
        pbar = trange(cfg.num_train_episodes)
        for ep in pbar:
            success, R = trainer.run(Env, epoch=ep)
            if log:
                logger.log({"epoch": ep, "success": success, "R": R})
            reward = np.round(R, 2)
            pbar.set_description(f"{ep=}, {reward=}, {t_total=}")
            trainer.train()
            # Train

        # print()
    # Return results
    if log:
        logger.process("epoch")
        logger.show()
    return episode_returns  # , best_actions


if __name__ == "__main__":
    run_continuous_agent()
