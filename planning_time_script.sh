

mpirun -np 1 python planning_time.py --config-name=run_sst env='Quadcopter'
mpirun -np 1 python planning_time.py --config-name=run_hoot env='Quadcopter'

mpirun -np 1 python planning_time.py --config-name=run_HER env='Quadcopter'

mpirun -np 1 python planning_time.py --config-name=run_continuous_optional_HER_training env='Quadcopter'
mpirun -np 1 python planning_time.py --config-name=run_volumeMCTS_optional_HER_training env='Quadcopter'
