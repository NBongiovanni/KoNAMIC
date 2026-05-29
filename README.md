# KoNAMIC

**Koopman-based Neural Architecture for Model Identification and Control**

KoNAMIC is a research codebase for learning Koopman-inspired dynamical models from sensor and visual data, with a focus on quadrotor identification and control. The learned latent dynamics are used both for open-loop prediction and for closed-loop control with MPC or baseline controllers.

The repository accompanies the experiments presented in the manuscript. It is primarily intended for transparency, reproducibility, and future extension of the thesis work.

## Repository status

The main training, dataset generation, simulation, and visualization pipelines are provided through dedicated entry points located in the `entrypoints/` directory. Some auxiliary scripts are still being cleaned up after recent refactoring and may require minor adjustments before being executed out of the box.

In particular:

- user-facing workflows should be launched from the scripts in `entrypoints/`;
- configuration is handled through YAML files and command-line overrides;
- several internal modules contain a file named `cli.py`, but these files are parser/helper modules and are not meant to be executed directly;
- model training is handled by a single entry point, `entrypoints/training/train_model.py`, for both `sensor` and `vision` modalities.

## Main workflows

The codebase is organized around the following main workflows.

| Workflow | Entry point                                              | Purpose                                                                             |
|---|----------------------------------------------------------|-------------------------------------------------------------------------------------|
| Sensor dataset generation | `entrypoints/data_generation/generate_sensor_dataset.py` | Generate sensor-based trajectories and save them as `.npz` datasets.                |
| Vision dataset generation | `entrypoints/data_generation/generate_vision_dataset.py` | Render and preprocess image datasets from previously generated sensor trajectories. |
| Model training | `entrypoints/train_model.py`                    | Train a Koopman-inspired model from sensor or visual data.                          |
| Open-loop evaluation | `entrypoints/open_loop/run.py`                           | Evaluate a trained model by rolling it out without feedback control.                |
| Open-loop comparison | `entrypoints/open_loop/compare.py`                       | Overlay rollouts from several trained models using a YAML preset.                   |
| Closed-loop simulation | `entrypoints/closed_loop/run.py`                         | Run closed-loop simulations with Koopman MPC or baseline controllers.               |
| Closed-loop comparison | `entrypoints/closed_loop/compare.py`                     | Compare several closed-loop runs using a YAML preset.                               |

## Installation

Clone the repository and create a Python environment:
```bash
git clone https://github.com/NBongiovanni/KoNAMIC.git
cd KoNAMIC
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```
> Note: depending on your system and CUDA version, you may prefer to install PyTorch separately following the official PyTorch instructions before installing the remaining dependencies.

## Dataset generation

Datasets are generated in two steps. First, sensor trajectories are simulated and stored as `.npz` files. Then, for visual experiments, these trajectories are converted into image datasets.

### Sensor datasets

Sensor datasets of a quadrotor (2d or 3d) can be generated with:

```bash
python entrypoints/data_generation/generate_sensor_dataset.py \
  --sensor_data_config configs/data_generation/sensor_2d.yaml \
  --modality sensor \
  --drone-dim 2
```

The script loads the sensor dataset configuration, builds the corresponding drone and plant, runs the configured controller, and writes the generated trajectories to a timestamped directory under `datasets/<drone_dim>d/`.

### Vision datasets

Vision datasets of a quadrotor are generated from an existing sensor dataset:

```bash
python entrypoints/data_generation/generate_vision_dataset.py \
  --vision_data_config configs/data_generation/vision_2d.yaml \
  --modality vision \
  --drone-dim 2 \
  --data_generation-stamp <sensor_dataset_stamp>
```

The vision pipeline loads the previously generated sensor trajectories, renders raw images, and preprocesses them into memory-mapped arrays for training. The generated files are stored in the dataset directory associated with `--dataset-stamp`.

## Training a model

A single training entry point is used for both sensor and visual data:

```bash
python entrypoints/training/train_model.py \
  --modality <sensor_or_vision> \
  --drone_dim 2 \
  --dynamics linear \
  --config <config_name> \
  --id <run_id> \
  --seed 0
```

For example, to train a model from sensor data:

```bash
python entrypoints/training/train_model.py \
  --modality sensor \
  --drone_dim 2 \
  --dynamics linear \
  --config sensor_2d \
  --id <run_id> \
  --seed 0
```

To train a model from visual data:

```bash
python entrypoints/training/train_model.py \
  --modality vision \
  --drone_dim 2 \
  --dynamics linear \
  --config vision_2d \
  --id <run_id> \
  --seed 0 \
  --data_generation-stamp <vision_dataset_stamp>
```

The selected `--modality` determines how the training data are prepared internally. With `sensor`, the entry point loads the state/input dataset directly. With `vision`, it loads the corresponding preprocessed image dataset together with the associated state/input scalers.

Typical arguments are:

| Argument | Description |
|---|---|
| `--modality` | Input modality: usually `sensor` or `vision`. |
| `--drone_dim` / `--drone-dim` | Drone model dimension, for example `2` or `3`, depending on the parser. |
| `--dynamics` | Latent dynamics structure, for example `linear`, `bilinear`, or another implemented model type. |
| `--config` | Name of the YAML training configuration to load, for example `sensor_2d` or `vision_2d`. |
| `--id` | Identifier used to name the training run. |
| `--seed` | Random seed. |
| `--dataset-stamp` | Dataset timestamp used when training from a previously generated vision dataset. |
| `--geom_losses` / `--no-geom_losses` | Enable or disable geometric auxiliary losses when supported. |
| `--state_in_z` / `--no-state_in_z` | Enable or disable inclusion of the state in the latent representation when supported. |

Training configurations are loaded according to the selected modality and drone dimension. The final resolved configuration is saved with the run outputs.

## Open-loop evaluation

To evaluate one trained model in open loop:

```bash
python entrypoints/open_loop/run.py \
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
| `--seed` | Random seed used for rollout selection. |

The script loads the selected trained case, runs open-loop rollouts, and saves the evaluation outputs and figures under the corresponding run directory.

To compare several open-loop results, use a preset from `configs/figures/open_loop.yaml`:

```bash
python entrypoints/open_loop/compare.py --preset <preset_name>
```

or with a custom preset file:

```bash
python entrypoints/open_loop/compare.py \
  --preset-file configs/figures/open_loop.yaml \
  --preset <preset_name>
  --dt <dt>
```

## Closed-loop simulation

To run a closed-loop simulation:

```bash
python entrypoints/closed_loop/run.py \
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

The script builds the selected controller and plant, runs the configured closed-loop simulations, and then generates the associated plots.

To compare several closed-loop simulations, use a preset from `configs/figures/control.yaml`:

```bash
python entrypoints/closed_loop/compare.py --preset <preset_name>
```

or:

```bash
python entrypoints/closed_loop/compare.py \
  --preset-file configs/figures/control.yaml \
  --preset <preset_name>
  --dt <dt>
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

Command-line arguments are used to select the experiment setup and to override selected configuration options. The resolved configuration is saved next to the generated results whenever applicable.

## Outputs

Depending on the workflow, the code writes:

- generated sensor trajectories;
- rendered and preprocessed image datasets;
- trained model checkpoints;
- resolved configuration files;
- scalers used for state and input normalization;
- open-loop prediction results;
- closed-loop simulation results;
- generated figures and comparison plots.

Outputs are stored in timestamped run directories to avoid overwriting previous experiments.

## Notes for developers

Several subpackages define a local `cli.py` file. This is intentional: each `cli.py` contains the argument parser for a specific pipeline. To avoid ambiguity, users should launch the scripts in `entrypoints/` rather than invoking `cli.py` files directly.

When adding a new workflow, prefer the following pattern:

1. define the parser in the relevant pipeline module;
2. expose a clear script in `entrypoints/`;
3. document the minimal command in this README;
4. save the resolved configuration next to the generated results.

## Citation

This repository accompanies the following paper:

N. Bongiovanni, B. Mavkov, R. Martins, and G. Allibert, “Identification and Control of a Planar Quadrotor from Visual Data Using Koopman Representations,” in *International Conference on Unmanned Aircraft Systems (ICUAS)*, 2026.

If you use this code or build upon this work, please cite the paper above.
