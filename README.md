=================================================================

# Copyright and License

All Code in this repository - unless otherwise stated in local license or code headers is

Copyright 2024 Max Planck Institute for Intelligent Systems

Licensed under the terms of the GNU General Public Licence (GPL) v3 or higher.
See: https://www.gnu.org/licenses/gpl-3.0.en.html



# installation
- create workspace
```console
mkdir MultitaskRL
cd MultitaskRL
```

- setup isaac-gym 
1. download isaac-gym from https://developer.nvidia.com/isaac-gym
2. extract isaac-gym to the workspace 
3. create conda environment and install the dependencies 
```console
bash IsaacGym_Preview_4_Package/isaacgym/create_conda_env_rlgpu.sh 
conda activate rlgpu
```

- clone this repository to the workspace and install dependencies
```console
git clone https://github.com/robot-perception-group/adaptive_agent.git
pip install -r adaptive_agent/requirements.txt
```

# Run Agent: 
- enter the RL workspace
```console
cd adaptive_agent/
```

- start learning in 25 environments with agents available: SAC, COMP, RMACOMP, PID
```
python run.py agent=COMP wandb_log=False env=BlimpRand env.num_envs=25  env.sim.headless=False
```

- The experiments are stored in the sweep folder. For example, hyperparameter tuning for the composition agent
```console
wandb sweep sweep/comp_hyper.yml
```
The experimental results are gathered in Wandb. 
