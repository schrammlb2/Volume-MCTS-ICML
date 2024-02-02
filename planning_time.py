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
# name="external_training_HER"

from get_az_args import get_az_args

from alphazero.agent.agents import *
from alphazero.search.mcts import *

import omegaconf


@hydra.main(config_path="config", config_name=f"run_{name}")
def run_continuous_agent(cfg):

    # cfg.env = omegaconf.OmegaConf.load("../../../config/env/FetchReach.yaml").env
    # print(f"Running {name}")
    # # print(cfg)
    # # return
    # train_her = f"_her:{cfg.train_her}"
    # train_az = f"_az:{cfg.train_az}"
    # standard_her = f"_full_traj_her:{cfg.standard_her}"
    # print(train_her)
    # print(train_az)
    # print(standard_her)


    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []
    # Environments
    # Env = make_game(cfg.game)
    # Env = make_game(global_config.environment_name)
    Env = make_game(cfg.env.game)
    Env.reset()


    # filename = f"PlanningTime/{cfg.env.game}/" + name + train_her + train_az + standard_her
    # print(f"Running {name}")
    print(f"Running {cfg.name}")
    filename = f"PlanningTime/{cfg.env.game}/" + cfg.name
    logger = Logger(["epoch", "R"], env_name=cfg.env.game, filename=filename)

    # set seeds
    np.random.seed(cfg.seed)
    # Env.seed(cfg.seed)

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

    # pbar = trange(cfg.num_train_episodes)
    # n_runs = 4
    n_runs = 60#cfg.env.n_trials_for_search_time
    n_planning_increments = 20
    # pbar = trange(1, n_planning_increments + 1)
    # pbar = trange(cfg.num_train_episodes)

    #mini version
    # n_runs = 4#cfg.env.n_trials_for_search_time
    # n_planning_increments = 5

    pbar = trange(n_planning_increments )

    n_rollouts = cfg.env.n_rollouts_per_step
    # rollout_list = [int(ep/n_planning_increments*n_rollouts) for ep in range(cfg.num_train_episodes)]

    for ep in pbar:
        cfg.mcts.n_rollouts = int(ep/n_planning_increments*n_rollouts)#*20
        rollouts = cfg.mcts.n_rollouts

        # her_agent.learn()
        if callable(Env.unwrapped):
            unwrapped_env = Env.unwrapped()
        else:
            unwrapped_env = Env.unwrapped

        unwrapped_env = StepAPICompatibility(unwrapped_env, output_truncation_bool=False)
        unwrapped_env = ResetCompatibilityWrapper(unwrapped_env)

        # trainer = hydra.utils.instantiate(cfg.trainer)

        # cfg.trainer.Env = Env
        # cfg.trainer.unwrapped_env = unwrapped_env
        # trainer = hydra.utils.instantiate(cfg.trainer)
        # from one_shot_trainer import OneShotTrainer
        # trainer = OneShotTrainer(cfg=cfg, Env=Env, unwrapped_env=unwrapped_env)
        trainer = hydra.utils.instantiate(cfg.trainer, 
                cfg=cfg, Env=Env, unwrapped_env=unwrapped_env)
        # trainer = trainer_constructor(cfg, Env, unwrapped_env)

        # pbar = trange(cfg.num_train_episodes)
        r_list = []
        success_list = []
        for run_number in range(n_runs):
            success, R = trainer.run(Env, epoch=ep)
            
            logger.log({"planning steps": rollouts*cfg.max_episode_length, "success": success, "R": R})
            
            r_list.append(R)
            success_list.append(success)

            reward_mean = np.round(sum(r_list)/len(r_list), 2)
            success_mean = np.round(sum(success_list)/len(success_list), 2)
            pbar.set_description(f"{ep=}, {run_number=}, {rollouts=}, {reward_mean=}, {success_mean=}")

            # info_dict["Episode reward"] = R
            # if isinstance(agent.loss, A0CLossTuned):
            #     info_dict["alpha"] = agent.loss.alpha.detach().cpu().item()

            # if run is not None:
            #     run.log(
            #         info_dict,
            #         step=ep,
            #     )

        # print()
    # Return results
    # logger.process("planning steps per environment step")
    logger.process("planning steps")
    # logger.show()
    return episode_returns  # , best_actions


if __name__ == "__main__":
    run_continuous_agent()
