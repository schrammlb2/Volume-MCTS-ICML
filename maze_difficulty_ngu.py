from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import trange
# import wandb
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.agent.losses import A0CLossTuned, A0CLoss, RPOLoss
from alphazero.helpers import check_space, store_actions
from rl.make_game import make_game
import global_config
from utils.logger import Logger
from alphazero.search.kd_tree import KDTree

wifi = False
# wifi = True
# name="continuous"
name="never_give_up"

mean = lambda lst: sum(lst)/len(lst)
#The idea of this variant is to have the KDtree form an consistent estimator 
#of the value by taking the value of a node from halfway down the tree

@hydra.main(config_path="config", config_name=f"run_{name}")
def run_agent(cfg: DictConfig):
    env_base = global_config.env_base
    print(f"Running {name} on environment {env_base}")
    logger = Logger(["maze_size", "R"], filename=name + "", env_name=env_base)
    R_max = -np.inf
    best_actions = None
    actions_list = []


    pbar = trange(2, global_config.max_size)
    # pbar.set_description(f"{0}, {0}, {0}")
    buffer = hydra.utils.instantiate(cfg.buffer)
    for maze_size in pbar:
        t_total = 0  # total steps
        n_seeds = 5
        for training_run in range(3):
            full_episode_returns = []  # storage
            full_success_log = []
            cfg.game = f"{env_base}-{maze_size}"
            # Environments
            Env = make_game(cfg.game)

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
            cfg.max_episode_length = global_config.max_episode_length
            cfg.mcts.n_rollouts = global_config.n_rollouts_per_step

            agent = hydra.utils.instantiate(cfg.agent)
            # print(agent.nn.bounds)

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

            if isinstance(agent.loss, A0CLossTuned):
                config.update(
                    {
                        "Target entropy": -cfg.agent.loss_cfg.action_dim,
                        "Loss lr": 0.001,
                        "Loss type": "A0C loss tuned",
                    }
                )
            elif isinstance(agent.loss, A0CLoss):
                config.update(
                    {
                        "Entropy coeff [alpha]": cfg.agent.loss_cfg.alpha,
                        "Loss type": "A0C loss untuned",
                    }
                )

            n_runs = global_config.n_runs
            episode_returns = []  # storage
            success_log = []
            # pbar = trange(cfg.num_train_episodes)
            buffer.clear()
            agent = hydra.utils.instantiate(cfg.agent)
            for run_number in range(n_runs):
            # for _ in range(1):

                if global_config.using_gymnasium:
                    state = Env.reset()[0]
                else:
                    state = Env.reset()
                R = 0.0  # Total return counter

                kd_tree = KDTree()
                kd_tree.add(state.tolist(), None)
                agent.reset_mcts(root_state=state)
                agent.epoch = run_number + 1
                agent.set_kd_tree(kd_tree)
                for t in range(cfg.max_episode_length):
                    # MCTS step
                    # run mcts and extract the root output
                    action, s, actions, counts, Qs, V = agent.act(Env=Env)
                    if len(actions.shape) == 1:
                        actions.reshape(-1, 1)
                    # import ipdb
                    # ipdb.set_trace()
                    buffer.store((s, actions, counts, Qs, V))

                    # Make the true step
                    action = action.reshape(-1)
                    state, step_reward, terminal, _ = Env.step(action)
                    kd_tree.add(state.tolist(), None)
                    actions_list.append(action)

                    R += step_reward
                    t_total += (
                        agent.n_rollouts  # total number of environment steps (counts the mcts steps)
                    )

                    if terminal or t == cfg.max_episode_length - 1:
                        R += step_reward*(cfg.max_episode_length - 1 - t)
                        if R_max < R:
                            # actions_list.insert(0, Env.seed())
                            best_actions = deepcopy(actions_list)
                            R_max = R
                            store_actions(cfg.game, best_actions)
                        actions_list.clear()
                        break
                    else:
                        # reset the mcts as we can't reuse the tree
                        agent.reset_mcts(root_state=state)
                # store the total episode return
                episode_returns.append(R)
                success = 1 if R > 0 else 0
                success_log.append(success)
                    

                info_dict = agent.train(buffer)

            mean_r = mean(episode_returns)
            mean_success = mean(success_log)
            full_episode_returns.append(mean_r)
            full_success_log.append(mean_success)
            logger.log({"maze_size": maze_size, "R": mean_r, "success": mean_success})
            reward = np.round(mean_r, 2)
        

        R = mean(full_episode_returns)
        success = mean(full_success_log)
        pbar.set_description(f"{maze_size=}, {R=}, {success=}, {t_total=}")
    # Return results
    logger.process("maze_size")
    logger.show()
    return episode_returns  # , best_actions


if __name__ == "__main__":
    run_agent()
