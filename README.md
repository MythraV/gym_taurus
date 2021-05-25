The gym environment for Taurus

# gym-Taurus

gym environment for Taurus robot using CoppeliaSim simulator.

## Requirements

Please install the following requirements before using this environment. It is recommended to do this in a virtual environment..

[VREP](https://www.coppeliarobotics.com/previousVersions) version 3.6.2
[PyRep](https://github.com/MythraV/PyRep.git) version compatible with VREP 3.6
[gym](https://github.com/openai/gym.git)

## RL setup


# Installation

```bash
git clone https://github.com/MythraV/gym_taurus.git
cd gym-taurus
pip install -e .
```

# Example Usage
Test environment with gym
```bash
python
import gym
env = gym.make('gym_taurus:taurus-deb-v0')
```
Running with openai/baselines or stable_baselines
```bash
python -m baselines.run --alg=ppo2 --env=gym_taurus:taurus-deb-v0 --network=mlp --num_timesteps=2e7 --num_env=6 --save_path=/media/crl/DATA/Datasets/RLmodels/taurus --save_interval=1e5
```
