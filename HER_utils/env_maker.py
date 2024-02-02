import gym
from gym.wrappers.time_limit import TimeLimit
from HER_utils.action_randomness_wrapper import ActionRandomnessWrapper

# import gymnasium

def make_env(args):
    # create the ddpg_agent
    use_gymnasium = False
    use_gymnasium = True
    env_name = args.env_name
    if "Quadcopter" in env_name:        
        import gym
        from rl.envs.mobile_mujoco_environments.factory import MushrEnvironmentFactory
        env_factory = MushrEnvironmentFactory(
            max_speed=0.5,
            max_steering_angle=0.5,
            max_steps=100,
            prop_steps=10,
            goal_limits=[0, 5],
            with_obstacles=True,
        )
        env_factory.register_environments_with_position_goals()
        # env = gym.make("X2ObsEnv-v0")
        from rl.envs.mobile_mujoco_environments.envs.quadrotor_env import QuadrotorReachEnv
        env = QuadrotorReachEnv(max_steps=30, noisy=False, use_obs=True,
                 use_orientation=False, noise_scale=0.01,
                 return_full_trajectory=False, max_speed=1.0, prop_steps=100)
        # import ipdb
        # ipdb.set_trace()
        max_episode_steps =  env._max_episode_steps
        from gymnasium.wrappers import StepAPICompatibility
        from rl.wrappers import ResetCompatibilityWrapper
        import gymnasium
        from gymnasium.wrappers.time_limit import TimeLimit as gymnasiumTimeLimit
        # env = gymnasiumTimeLimit(env, max_episode_steps=max_steps)
        env = StepAPICompatibility(env, output_truncation_bool=False)
        env = ResetCompatibilityWrapper(env)
        env._max_episode_steps = max_episode_steps
        return env
    if use_gymnasium:
        if "RotationFetch" in env_name:
            import gymnasium
            import gymnasium_robotics
            from gymnasium.wrappers.time_limit import TimeLimit as gymnasiumTimeLimit
            if "Reach" in env_name:
                from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.reach import RotationMujocoFetchReachEnv as FetchReachEnv
                env = FetchReachEnv()
            elif "Push" in env_name:
                from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.push import RotationMujocoFetchPushEnv as FetchPushEnv
                env = FetchPushEnv()
            elif "Slide" in env_name:
                from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.slide import RotationMujocoFetchSlideEnv as FetchSlideEnv
                env = FetchSlideEnv()
            elif "PickAndPlace" in env_name:
                from rl.envs.pomp.example_problems.gymnasium_robotics_local.gymnasium_robotics.envs.fetch.pick_and_place import RotationMujocoFetchPickAndPlaceEnv as FetchPickAndPlaceEnv
                env = FetchPickAndPlaceEnv()

            # env = gymnasium.vector.make('FetchReach-v3', num_envs=3)
            max_steps = 50
            import ipdb
            ipdb.set_trace()
            env = gymnasiumTimeLimit(env, max_episode_steps=max_steps)
            from gymnasium.wrappers import StepAPICompatibility
            from rl.wrappers import ResetCompatibilityWrapper
            env = StepAPICompatibility(env, output_truncation_bool=False)
            env = ResetCompatibilityWrapper(env)
            env._max_episode_steps = 50
            return env
        elif "Fetch" in env_name:
            from rl.envs.pomp.example_problems.robotics.fetch.reach import FetchReachEnv
            from rl.envs.pomp.example_problems.robotics.fetch.push import FetchPushEnv
            from rl.envs.pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
            from rl.envs.pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
            if "FetchReach" in env_name:
                env = TimeLimit(FetchReachEnv(), max_episode_steps=50)
            elif "FetchPush" in env_name:
                env = TimeLimit(FetchPushEnv(), max_episode_steps=50)
            elif "FetchSlide" in env_name:
                env = TimeLimit(FetchSlideEnv(), max_episode_steps=50)
            elif "FetchPickAndPlace" in env_name:
                env = TimeLimit(FetchPickAndPlaceEnv(), max_episode_steps=50)
        else:
            import gymnasium
            import gymnasium_robotics
            env = gymnasium.make(args.env_name)
            from gymnasium.wrappers import StepAPICompatibility
            from rl.wrappers import ResetCompatibilityWrapper
            env = StepAPICompatibility(env, output_truncation_bool=False)
            env = ResetCompatibilityWrapper(env)
            env._max_episode_steps = 50
            return env

    elif (not use_gymnasium) and "Fetch" in args.env_name:
        rotation=("Rotation" in args.env_name)
        from rl.envs.pomp.example_problems.robotics.fetch.reach import FetchReachEnv
        from rl.envs.pomp.example_problems.robotics.fetch.push import FetchPushEnv
        from rl.envs.pomp.example_problems.robotics.fetch.slide import FetchSlideEnv
        from rl.envs.pomp.example_problems.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
        if "FetchReach" in args.env_name:
            env = TimeLimit(FetchReachEnv(rotation=rotation), max_episode_steps=50)
        elif "FetchPush" in args.env_name:
            env = TimeLimit(FetchPushEnv(rotation=rotation), max_episode_steps=50)
        elif "FetchSlide" in args.env_name:
            env = TimeLimit(FetchSlideEnv(rotation=rotation), max_episode_steps=50)
        elif "FetchPickAndPlace" in args.env_name:
            env = TimeLimit(FetchPickAndPlaceEnv(rotation=rotation), max_episode_steps=50)

    elif args.env_name == "Throwing":
        env = TimeLimit(make_throwing_env(), max_episode_steps=20)
    elif "CarEnvironment" in args.env_name:
        env = TimeLimit(CarEnvironment(), max_episode_steps=50)
    elif "Torus" in args.env_name:
        freeze = "Freeze" in args.env_name or "freeze" in args.env_name
        if freeze: 
            n = args.env_name[len("TorusFreeze"):]
        else: 
            n = args.env_name[len("Torus"):]
        try: 
            dimension = int(n)
        except:
            print("Could not parse dimension. Using n=2")
            dimension=2
        print(f"Dimension = {dimension}")
        print(f"Freeze = {freeze}")
        env = TimeLimit(Torus(dimension, freeze), max_episode_steps=50)
    else:
        # env = gymnasium.make(args.env_name, max_episode_steps=50)
        # from gymnasium.wrappers import StepAPICompatibility
        # from HER_utils.reset_wrapper import ResetWrapper, StepNumWrapper
        # env = StepAPICompatibility(env, output_truncation_bool=False)
        # env = ResetWrapper(env)
        # env = StepNumWrapper(env)
        env = gym.make(args.env_name)


    env = ActionRandomnessWrapper(env, args.action_noise)

    return env
