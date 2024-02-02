import argparse

def alphazero_get_args():
	parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', type=str, default='local_maze-7', help='the environment name')
    parser.add_argument('--max-episode-length', type=int, default=50, help='number of timesteps allowed per episode')
    parser.add_argument('--n-rollouts-per-step', type=int, default=50, help='number of rollours per search tree (in closed loop search)')
    parser.add_argument('--n-epochs', type=int, default=1, help='the number of epochs to train the agent')

    args = parser.parse_args()
