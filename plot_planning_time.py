
from utils.logger import Logger

# env_name = "PlanningTime/Quadcopter"
# env_name = "PlanningTime/FetchReach"
smoothing = 2
x_var = "planning steps"
# logger = Logger([x_var, "R"], env_name=env_name)
logger = Logger([x_var, "R"], env_name="Quadcopter", relative_directory="PlanningTime/")
logger.plot_all(x_var, smoothing=smoothing)
logger = Logger([x_var, "success"], env_name="Quadcopter", relative_directory="PlanningTime/")
# logger = Logger([x_var, "R"], env_name="FetchReach", relative_directory="PlanningTime/")
logger.plot_all(x_var, smoothing=smoothing)