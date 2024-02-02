from copy import deepcopy
from datetime import datetime
import numpy as np
from tqdm import trange
import wandb
import hydra
from omegaconf.dictconfig import DictConfig

from alphazero.agent.losses import A0CLossTuned, A0CLoss
from alphazero.helpers import check_space, store_actions
from rl.make_game import make_game
import global_config

from rl.envs.pomp.visualizer import run_gym_visualizer

wifi = False
# wifi = True

name="continuous"
# name="rpo_advantage"
# name="rpo_whole_tree"
# name="volume_2"

# name="multi_step"
# name="one_shot_2"
name="open_loop_continuous"
# name="ssmcts"

@hydra.main(config_path="config", config_name=f"run_{name}")
def run_visualization(cfg: DictConfig):
    cfg.game = f"local_maze-7"
    episode_returns = []  # storage
    R_max = -np.inf
    best_actions = None
    actions_list = []
    # Environments
    Env = make_game(cfg.game, run_maze=True)

    # set seeds
    np.random.seed(cfg.seed)
    # Env.seed(cfg.seed)

    buffer = hydra.utils.instantiate(cfg.buffer)

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
    # cfg.policy.action_bound = Env.action_space.high.tolist()
    # cfg.policy.action_bound = None

    if "volume" in name or "one_shot" in name or "ssmcts": 
        lo = [float(i) for i in Env.observation_space.low]
        hi = [float(i) for i in Env.observation_space.high]
        cfg.mcts.observation_bounds = lo, hi

    agent = hydra.utils.instantiate(cfg.agent)
    print(agent.nn.bounds)

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

    cfg.num_train_episodes = 0
    pbar = trange(cfg.num_train_episodes)
    for ep in pbar:
        if global_config.using_gymnasium:
            state = Env.reset()[0]
        else:
            state = Env.reset()
        R = 0.0  # Total return counter

        agent.reset_mcts(root_state=state)
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
            actions_list.append(action)

            R += step_reward
            t_total += (
                agent.n_rollouts  # total number of environment steps (counts the mcts steps)
            )

            if terminal or t == cfg.max_episode_length - 1:
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

        # Train
        info_dict = agent.train(buffer)
        # agent.save_checkpoint(env=Env)

        info_dict["Episode reward"] = R
        if isinstance(agent.loss, A0CLossTuned):
            info_dict["alpha"] = agent.loss.alpha.detach().cpu().item()

        if run is not None:
            run.log(
                info_dict,
                step=ep,
            )

        reward = np.round(R, 2)
        pbar.set_description(f"{ep=}, {reward=}, {t_total=}")
    # Return results
    # return episode_returns  # , best_actions
    state = Env.reset()[0]
    agent.reset_mcts(state)
    agent.mcts.initialize_search()
    assert agent.mcts.root_node is not None
    agent.mcts.add_value_estimate(agent.mcts.root_node)
    agent.mcts.add_pw_action(agent.mcts.root_node)
    agent.mcts.Env = Env
    run_gym_visualizer(Env, agent.mcts)


if __name__ == "__main__":
    run_visualization()
