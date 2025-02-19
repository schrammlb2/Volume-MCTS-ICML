from omegaconf import OmegaConf

"""
Here are the param for the training

"""

def get_args():
    conf = OmegaConf.create(
    {
        'env_name': 'FetchReach_v3',
        'n_epochs': 10, 
        'n_cycles': 50,
        'n_batches':40,
        'save_interval': 5,
        'seed': 123,
        'num_workers':1,
        'replay_strategy': 'future',
    
        'clip_return': 50,
        'save_dir': 'saved_models/',
        'noise_eps':0.2,
        'random_eps': 0.3, 
        'buffer_size': int(1e6), 
        'replay_k': 4,
        'clip_obs': 200,
        'batch_size': 256,
        
        'entropy_regularization': 0.01, 

        'gamma': 0.98,
        'action_l2': 1, 

        'lr_actor': 0.001,
        'lr_critic': 0.001, 
        'polyak': 0.95,
        'n_test_rollouts': 40,
        'clip_range': 5,
        'demo_length': 20,
        'cuda': False,
        'p2p':False,
        'num_rollouts_per_mpi': 2,

        'non_terminal_goals': False,
        'off_goal': 0.1, 
        'action_noise': 0.,
        'two_goal': False,
        'apply_ratio': False,
        'ratio_offset': 0.1,
        'ratio_clip': 0.3,
    })

    return conf
