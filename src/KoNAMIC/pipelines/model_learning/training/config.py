from dataclasses import dataclass
from copy import deepcopy
from argparse import Namespace

from KoNAMIC.core import utils, drone


@dataclass(frozen=True)
class PredictionHorizon:
    train: int
    val: list[int]

    @classmethod
    def from_dataset_params(cls, dataset_params: dict) -> "PredictionHorizon":
        return cls(
            train=dataset_params["train"]["num_steps_pred"],
            val=[
                val_dataset["num_steps_pred"]
                for val_dataset in dataset_params["val_datasets"]
            ],
        )


@dataclass
class TrainingConfig:
    dataset_params: dict
    training_params: dict
    model_params: dict
    control_params: dict

    @property
    def prediction_horizon(self) -> PredictionHorizon:
        return PredictionHorizon.from_dataset_params(self.dataset_params)

    @classmethod
    def from_dict(cls, params: dict, control_params, dataset_params) -> "TrainingConfig":
        params = deepcopy(params)
        return cls(
            dataset_params=dataset_params,
            training_params=params["training_params"],
            model_params=params["model_params"],
            control_params=control_params,
        )

    @classmethod
    def load_base_config(cls, name: str, modality: str, drone_dim: int) -> "TrainingConfig":
        root = utils.find_project_root()

        training_config_path = root / "configs" / "training" / f"{name}.yaml"
        control_config_path = root / "configs" / "control" / f"knmpc_{modality}_{drone_dim}d_base.yaml"
        dataset_config_path = root / "configs" / "data" / f"{modality}_{drone_dim}d.yaml"

        params = utils.load_yaml(training_config_path)
        control_params = utils.load_yaml(control_config_path)
        dataset_params = utils.load_yaml(dataset_config_path)

        return cls.from_dict(
            params=params,
            control_params=control_params,
            dataset_params=dataset_params,
        )

    def to_dict(self) -> dict:
        return {
            "dataset_params": self.dataset_params,
            "training_params": self.training_params,
            "model_params": self.model_params,
            "control_params": self.control_params,
        }

    def sync_shared_params(self) -> None:
        drone_dim = self.model_params["drone_dim"]
        x_dim, u_dim, _ = drone.get_dimensions(drone_dim)
        self.model_params["z_dynamics"]["x_dim"] = x_dim
        self.model_params["z_dynamics"]["u_dim"] = u_dim

        self.model_params["auto_encoder"]["delay"] = self.dataset_params["delay"]
        self.control_params["dt"] = self.model_params["dt"]

    def apply_cli_options(self, args: Namespace) -> None:
        self.model_params["z_dynamics"]["model"] = args.dynamics
        if args.modality == "sensor":
            self.model_params["auto_encoder"]["include_state_in_z"] = args.state_in_z
            if args.state_in_z:
                self.model_params["z_dynamics"]["structured_AB"] = False
        self.training_params["seed"] = args.seed

    def define_paths(self, paths: utils.RunPaths) -> None:
        self.training_params["run_dir"] = paths.run_dir
        self.control_params["control_runs_dir"] = paths.closed_loop_eval_dir
        self.training_params["log_dir"] = paths.log_dir
        self.training_params["checkpoints_dir"] = paths.run_dir / "checkpoints"