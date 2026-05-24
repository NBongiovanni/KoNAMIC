# KoNAMIC

**Koopman-based Neural Architecture for Model Identification and Control**

KoNAMIC is a research codebase for learning Koopman-inspired dynamical models from sensor and visual data, with a focus on quadrotor identification and control. The learned latent dynamics are used both for open-loop prediction and for closed-loop control with MPC or baseline controllers.

The repository accompanies the experiments presented in the manuscript. It is primarily intended for transparency, reproducibility, and future extension of the thesis work.

## Repository status

The main training, simulation, and visualization pipelines are provided. Some auxiliary scripts are still being cleaned up after recent refactoring and may require minor adjustments before being executed out of the box.

In particular:

- the main user-facing workflows should be launched from the `run_*.py` entry points;
- configuration is handled through YAML files and command-line overrides;
- several internal modules contain a file named `cli.py`, but these files are parser/helper modules and are not meant to be executed directly.

## Main workflows

The codebase is organized around five main workflows.

| Workflow | Entry point | Purpose |
|---|---|---|
| Sensor dataset generation | `run_sensor_dataset_generation.py` | Generate sensor-based trajectories and save them as `.npz` datasets. |
| Vision dataset generation | `run_vision_dataset_generation.py` | Convert state/input trajectories into image-based datasets. |
| Model training | `run_train_model.py` | Train a Koopman-inspired model from sensor or visual data. |
| Open-loop evaluation | `run_open_loop.py` | Evaluate a trained model by rolling it out without feedback control. |
| Open-loop comparison | `run_open_loop_comparison.py` | Overlay rollouts from several trained models using a YAML preset. |
| Closed-loop simulation | `run_closed_loop.py` | Run closed-loop simulations with Koopman MPC or baseline controllers. |
| Closed-loop comparison | `run_closed_loop_comparison.py` | Compare several closed-loop runs using a YAML preset. |

## Installation

Clone the repository and install it from the project root:

```bash
git clone https://github.com/NBongiovanni/KoNAMIC.git
cd KoNAMIC
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If a `requirements.txt` or environment file is provided, install it before running the examples:

```bash
pip install -r requirements.txt
```

Closed-loop MPC experiments may require additional dependencies such as `acados` and its Python interface.

## Training a model

The main training entry point is:

```bash
python run_train_model.py \
  --modality vision \
  --drone_dim 2 \
  --dynamics linear \
  --config <config_name> \
  --id <run_id> \
  --seed 0
```

Typical arguments are:

| Argument | Description |
|---|---|
| `--modality` | Input modality: usually `sensor` or `vision`. |
| `--drone_dim` | Drone model dimension, for example `2` or `3`. |
| `--dynamics` | Latent dynamics structure, for example `linear`, `bilinear`, or another implemented model type. |
| `--config` | Name of the YAML configuration to load. |
| `--id` | Identifier used to name the training run. |
| `--seed` | Random seed. |
| `--geom_losses` / `--no-geom_losses` | Enable or disable geometric auxiliary losses when supported. |
| `--state_in_z` / `--no-state_in_z` | Enable or disable inclusion of the state in the latent representation when supported. |

Training configurations are loaded from the configuration directory according to the selected modality and drone dimension. The final resolved configuration is saved with the run outputs.

## Dataset generation

### Sensor datasets

Sensor datasets can be generated with:

```bash
python run_sensor_dataset_generation.py
```

This script currently uses a predefined YAML configuration for 3D sensor data. Adjust the corresponding file in `configs/data/` before launching a new dataset generation.

### Vision datasets

Vision datasets are generated from previously created state/input trajectories:

```bash
python run_vision_dataset_generation.py --drone_dim 2
```

The vision pipeline renders raw images and preprocesses them into memory-mapped arrays for training. The generated images are stored in the image dataset directory defined by the dataset configuration.

## Open-loop evaluation

To evaluate one trained model in open loop:

```bash
python run_open_loop.py \
  --modality vision \
  --caseid <case_id> \
  --num-steps 100 \
  --num-rollouts 30 \
  --phase val_2 \
  --seed 3
```

Useful arguments include:

| Argument | Description |
|---|---|
| `--modality` | Selects `sensor` or `vision` evaluation. |
| `--caseid` | Identifier of the stored model case to evaluate. |
| `--num-steps` | Number of prediction steps in each rollout. |
| `--num-rollouts` | Number of rollouts to render. |
| `--phase` | Dataset split used for evaluation, for example `val_2`. |
| `--render-vision` | Render image trajectories when supported. |
| `--snapshots` | Enable snapshot rendering. |

To compare several open-loop results, use a preset from `configs/figures/open_loop.yaml`:

```bash
python run_open_loop_comparison.py --preset <preset_name>
```

or with a custom preset file:

```bash
python run_open_loop_comparison.py \
  --preset-file configs/figures/open_loop.yaml \
  --preset <preset_name>
```

## Closed-loop simulation

To run a closed-loop simulation:

```bash
python run_closed_loop.py \
  --modality vision \
  --drone-dim 2 \
  --controller-type koopman_mpc \
  --case-id <case_id> \
  --seed 3
```

Supported controller types include:

- `koopman_mpc`, for control based on a learned Koopman model;
- `pid`, for baseline PID control;
- `lqr`, when available for the selected setup.

To compare several closed-loop simulations, use a preset from `configs/figures/control.yaml`:

```bash
python run_closed_loop_comparison.py --preset <preset_name>
```

or:

```bash
python run_closed_loop_comparison.py \
  --preset-file configs/figures/control.yaml \
  --preset <preset_name>
```

## Configuration files

Most experiments are controlled by YAML configuration files. They define, among other things:

- dataset paths and dataset versions;
- model architecture and latent dimension;
- latent dynamics type;
- loss weights;
- training hyperparameters;
- prediction horizon;
- controller and MPC parameters;
- figure-generation presets.

Command-line arguments are used to select the experiment setup and to override selected configuration options.

## Outputs

Depending on the workflow, the code writes:

- trained model checkpoints;
- resolved configuration files;
- scalers used for state and input normalization;
- open-loop prediction results;
- closed-loop simulation results;
- generated figures and comparison plots.

Outputs are stored in timestamped run directories to avoid overwriting previous experiments.

## Notes for developers

Several subpackages define a local `cli.py` file. This is intentional: each `cli.py` contains the argument parser for a specific pipeline. To avoid ambiguity, users should launch the top-level `run_*.py` scripts rather than invoking `cli.py` files directly.

When adding a new workflow, prefer the following pattern:

1. define the parser in the relevant pipeline module;
2. expose a clear `run_*.py` entry point;
3. document the minimal command in this README;
4. save the resolved configuration next to the generated results.

## Citation

If this code is useful for your work, please cite the corresponding manuscript or thesis once available.