The code in this project is based off implementations from the following repos: 
    https://github.com/tmoer/a0c
    https://github.com/krishauser/pyOptimalMotionPlanning
    https://github.com/TianhongDai/hindsight-experience-replay
    https://github.com/schrammlb2/USHER\_Implementation
    https://github.com/xizeroplus/POLY-HOOT


To run, first create an environment from the env.yml file

To test the Maze experiments, go to global_config.py and set the "env" variable to either "dubins" or "geometric". Then run maze_training_script.sh. You can plot the results with plot_maze_size.py

To test the Quadcopter experiments, run planning_time_script.sh.  You can plot the results with plot_planning_time.py



visualization.py produces the visualization seen in Figure 1. shows the progress of a selected algorithm in the maze environment. 