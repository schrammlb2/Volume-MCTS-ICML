
from utils.logger import Logger

# env_name = "PlanningTime/Quadcopter"
# env_name = "PlanningTime/FetchReach"
x_var = "maze size"
# logger = Logger([x_var, "R"], env_name=env_name)
logger = Logger([x_var, "R"], env_name="Training", relative_directory="geometric-local_maze/")
logger.plot_all(x_var, smoothing=0)
logger = Logger([x_var, "R"], env_name="No Training", relative_directory="geometric-local_maze/")
logger.plot_all(x_var, smoothing=0)
# logger = Logger([x_var, "R"], env_name="MazePlot", relative_directory="dubins-local_maze/")
# logger = Logger([x_var, "success"], env_name="Quadcopter", relative_directory="PlanningTime/")
# logger = Logger([x_var, "R"], env_name="FetchReach", relative_directory="PlanningTime/")