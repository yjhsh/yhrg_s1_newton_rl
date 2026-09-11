# YHRG S1 Newton RL

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Simulator](https://img.shields.io/badge/Newton-1.5.1-green.svg)](https://github.com/newton-physics/newton)
[![RL](https://img.shields.io/badge/rsl__rl-4.0.1-orange.svg)](https://github.com/leggedrobotics/rsl_rl)

**Reinforcement-learning framework for non-prehensile robotic pushing with the YHRG S1 manipulator, built on the [Newton](https://github.com/newton-physics/newton) simulator (Warp).**

> This project is a port of [**yhrg_s1_genesis_rl**](https://github.com/yjhsh/yhrg_s1_genesis_rl) — the original implementation built on the [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator — to the **Newton** simulator. 
>
> For more details about **YHRG S1**, please refer to the official [S1_SDK](https://github.com/YHRG-Robotics/S1_SDK).


## Environment

Python 3.12, CUDA 12.6. Core Python dependencies:

```
newton==1.5.1
warp-lang==1.17.0
rsl-rl-lib==4.0.1
torch==2.7.0+cu126
pytorch3d==0.7.9
```


## Usage

Train:

```bash
python rl_integration.py --num_envs 4096 --max_iterations 5000
```

Evaluate:

```bash
python rl_integration.py --eval -e examples1 --checkpoint model_4999.pt --num_envs 1 --num_episodes 5 --vis
```

Checkpoints and TensorBoard logs are written to `logs/<exp_name>/` — the trained
example above lives in `logs/examples1/`.


### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp_name`, `-e` | `yhrg_s1_push` | Experiment name (defines the log/checkpoint subfolder). |
| `--num_envs`, `-B` | `16` | Number of parallel simulation environments. |
| `--max_iterations` | `13000` | PPO iterations (training) / unused in eval. |
| `--vis` | `False` | Open the Newton viewer. |
| `--eval` | `False` | Run evaluation instead of training. |
| `--checkpoint` | `None` | Checkpoint (`*.pt`) under `log_dir/exp_name/` for evaluation. |
| `--num_episodes` | `10` | Number of evaluation episodes. |
| `--log_dir` | `logs` | Root directory for logs and checkpoints. |

---

## Project Structure

```
yhrg_newton_genesis_rl/
├── run_newton.sh            # CUDA 12.6 environment wrapper
├── rl_integration.py        # RL training/evaluation entry point (rsl_rl glue)
├── rl_push_env.py           # PushEnv: scene, observations, actions, rewards
├── config/                  # Scene/robot configuration dataclasses
├── model/                   # Robot config, loaders, point-cloud, state
├── controller/              # Task-space controller, IK solver
├── view/                    # Simulator proxy and simulation flow handlers
├── logs/                    # Training checkpoints and TensorBoard logs
└── asset/                   # URDF / mesh / texture assets
```


## Acknowledgments

- **[Newton](https://github.com/newton-physics/newton)** — The physics simulator that powers all simulation and rendering (via the Warp kernel toolkit).
- **[rsl_rl](https://github.com/leggedrobotics/rsl_rl)** — The PPO implementation used for on-policy training.
- **[shifu](https://github.com/42jaylonw/shifu)** - RL reward design.
- The **YHRG S1** manipulator hardware/URDF model.
