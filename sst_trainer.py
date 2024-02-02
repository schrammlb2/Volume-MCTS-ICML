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
# name="continuous_external_training_HER"
name="external_training_HER"

from get_az_args import get_az_args

from alphazero.agent.agents import *
from alphazero.search.mcts import *

from trainer import Trainer

# from psst.pomp.example_problems.generic_gym_wrapper import GymEnvWrapper
# from psst.pomp.planners.rrtstarplanner import StableSparseRRT
# from psst.pomp.planners import test
# from psst.pomp.spaces.objectives import TimeLengthObjectiveFunction
# from psst.pomp.planners.problem import PlanningProblem

from rl.envs.pomp.example_problems.generic_gym_wrapper import GymEnvWrapper
from rl.envs.pomp.planners.rrtstarplanner import StableSparseRRT
from rl.envs.pomp.planners import test
from rl.envs.pomp.spaces.objectives import TimeLengthObjectiveFunction
from rl.envs.pomp.planners.problem import PlanningProblem


import gym

from gymnasium.wrappers import StepAPICompatibility
from rl.wrappers import ResetCompatibilityWrapper
from rl.SaveStateWrapper import wrap_with_save_state_wrapper, SaveStateWrapper
# from psst.main import testPlannerDefault


def rewrap(Env, base_env): 
    if callable(base_env.unwrapped): env = base_env.unwrapped()
    else: env = base_env.unwrapped

    if isinstance(env, gymnasium.core.Env):
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = ResetCompatibilityWrapper(env)
    
    if isinstance(Env , SaveStateWrapper): 
        env = wrap_with_save_state_wrapper(env)
    return env

class SSTTrainer(Trainer):
    def __init__(self, cfg, Env, unwrapped_env):
        # self.problem = GymEnvWrapper(new_env)
        unwrapped_env = rewrap(Env, unwrapped_env)
        self.problem = GymEnvWrapper(unwrapped_env)
        self.cfg = cfg

        lo = [float(i) for i in Env.observation_space.low]
        hi = [float(i) for i in Env.observation_space.high]
        cfg.mcts.observation_bounds = lo, hi


        self.rollouts = cfg.mcts.n_rollouts*cfg.env.max_episode_length


        self.args = cfg.her
        self.args.n_batches = 100
        self.args.n_epochs = 1

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

    def run(self, Env, epoch):
        if callable(Env.unwrapped):
            unwrapped_env = Env.unwrapped()
        else:
            unwrapped_env = Env.unwrapped

        unwrapped_env = rewrap(Env, unwrapped_env)
        
        # from rl.SaveStateWrapper import wrap_with_save_state_wrapper
        # from gymnasium.wrappers import StepAPICompatibility
        # env = StepAPICompatibility(env, output_truncation_bool=False)
        # new_env = wrap_with_save_state_wrapper(unwrapped_env)
        # self.problem = GymEnvWrapper(new_env)
        plannerParams = {
            # 'maxTime':self.rollouts, 
            'edgeCheckTolerance':0.01,
            'selectionRadius':.3,
            'witnessRadius':0.16
        }

        p = GymEnvWrapper(unwrapped_env)
        # objective = TimeObjectiveFunction()
        # controlSpace = p.controlSpace()
        # startState = p.startState()
        # goalSet = p.goalSet()
        # self.problem = PlanningProblem(controlSpace, startState, goalSet,
        #                        objective=objective)

        objective = TimeLengthObjectiveFunction()
        controlSpace = p.controlSpace()
        startState = p.startState()
        goalSet = p.goalSet()
        # self.problem = PlanningProblem(controlSpace, startState, goalSet,
        #                        objective=objective)
        prob_func = lambda: PlanningProblem(controlSpace, startState, goalSet,
                               objective=objective)
        cost, success = test.testPlanner(prob_func,1,self.rollouts + 1,
            None, 'stable-sparse-rrt', **plannerParams)

        if type(cost) is list: 
            import ipdb
            ipdb.set_trace()


        if cost > self.cfg.max_episode_length: 
            reward = 0#self.cfg.max_episode_length
        else: 
            reward = self.cfg.max_episode_length - cost

        return success, reward
        # return

    def train(self):  
        pass  
        # if self.cfg.train_her:
        #     if self.cfg.standard_her: 
        #         self.her_agent.perform_update(self.episode_data)
        #     else: 
        #         self.her_agent.perform_update(self.her_dict)
        # if self.cfg.train_az:
        #     info_dict = self.agent.train(self.buffer)



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

