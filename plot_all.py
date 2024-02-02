

from utils.logger import Logger

env_name = "RotationFetchReach"
env_name = "RotationFetchPush"

env_name = "dubins-local_maze"
env_name = "local_maze"
# env_name = "FetchReach-v3"
# env_name = "FetchReach-v2"
if "maze" in env_name:
	x_var = "maze_size"
	logger = Logger([x_var, "R"], env_name=env_name)
	logger.plot_all(x_var, include_training=True)
	logger.plot_all(x_var, include_training=False)
else: 
	x_var = "search_time"
	logger = Logger([x_var, "R"], env_name=env_name)
	logger.plot_all(x_var)
	logger = Logger([x_var, "success"], env_name=env_name)
	logger.plot_all(x_var)
	# logger.plot_all(x_var, include_training=False)
# logger = Logger([x_var, "success"], env_name=env_name)
# logger.plot_all(x_var)