global using_gymnasium
using_gymnasium = True
# using_gymnasium = False

if using_gymnasium: 
	import gymnasium as gym
else: 
	import gym
spaces=gym.spaces

env = "dubins"
# env = "geometric"
assert env in ["dubins", "geometric"]
if env == "dubins":
	environment_name = "dubins-local_maze-4"
	env_base = "dubins-local_maze"
	max_episode_length = 50
	n_rollouts_per_step = 100
	n_runs = 10
	max_size = 7
elif env == "geometric":
	environment_name = "local_maze-4"
	env_base = "local_maze"
	max_episode_length = 50
	n_rollouts_per_step = 100 #100
	n_runs = 10
	max_size = 10