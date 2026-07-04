import joblib

from KoNAMIC.pipelines.data_preparation.data_loaders import (
    DataLoaderSplit,
    PreparedDataLoaders,
)
from KoNAMIC.paths import DatasetPaths
from .config import SensorPreparationConfig
from .sensor_processor import SensorProcessor
from .sensor_loader import SensorLoader, SensorLoadSpec


class SensorBuilder:
    """Build sensor datasets and loaders for Koopman model training.

    SensorBuilder is the orchestration layer for sensor data preparation.
    It reads the raw dataset splits described by DatasetPaths and
    SensorPreparationConfig, applies the configured downsampling factor, and
    delegates feature scaling and sequence construction to SensorProcessor.

    The builder owns the resulting x and u scalers because they must be saved
    with the run and reused when evaluating a trained model. It also converts
    the processor raw dataloaders into PreparedDataLoaders, which carry the
    prediction horizon metadata expected by the training and evaluation code.

    Keep this class focused on wiring the preparation steps together. Loading
    raw arrays belongs to SensorLoader, numerical preprocessing belongs to
    SensorProcessor, and training and evaluation code should consume only the
    prepared loaders and saved scalers exposed here.
    """

    def __init__(
        self,
        dataset_paths: DatasetPaths,
        config: SensorPreparationConfig,
        system_dim: int,
        seed: int = 0
    ):
        ds_step = config.postprocessing.ds_step
        if ds_step < 1:
            raise ValueError(f"postprocessing.ds_step must be >= 1, got {ds_step}")

        split_specs = {
            "train": SensorLoadSpec(
                num_steps_loaded=None,
                num_traj_loaded=config.train.num_traj_loaded,
            ),
            **{
                val_cfg.split: SensorLoadSpec(
                    num_steps_loaded=None,
                    num_traj_loaded=val_cfg.num_traj_loaded,
                )
                for val_cfg in config.val_datasets
            },
        }

        self.loader = SensorLoader(
            dataset_paths=dataset_paths,
            system_dim=system_dim,
            split_specs=split_specs,
            downsample_factor=ds_step,
        )
        raw_data = self.loader.load_raw_sensor_data()
        batch_size = config.dataloader.batch_size

        self.processor = SensorProcessor(
            batch_size,
            config.train,
            config.val_datasets,
            raw_data,
            config.scaler,
            config.postprocessing.delay,
            seed,
        )
        self.processor.generate_u_scaler()
        self.processor.generate_x_scaler()

        self.processor.process_datasets()
        self.u_scaler = self.processor.u_scaler
        self.x_scaler = self.processor.x_scaler

        self.processor.build_raw_data_loader()
        self.data_loaders = self._build_prepared_data_loaders(config)

    def _build_prepared_data_loaders(
        self,
        params: SensorPreparationConfig,
    ) -> PreparedDataLoaders:
        raw_loaders = self.processor.raw_data_loaders
        train = DataLoaderSplit(
            split="train",
            name="train",
            loader=raw_loaders["train"],
            num_steps_pred=params.train.num_steps_pred,
        )
        validations = tuple(
            DataLoaderSplit(
                split=val_cfg.split,
                name=val_cfg.name,
                loader=raw_loaders[val_cfg.split],
                num_steps_pred=val_cfg.num_steps_pred,
            )
            for val_cfg in params.val_datasets
        )
        return PreparedDataLoaders(train=train, validations=validations)

    @property
    def processed(self):
        return self.processor.processed_datasets

    def save_scalers(self, run_dir) -> None:
        (run_dir / "scalers").mkdir(exist_ok=True)
        joblib.dump(self.u_scaler, run_dir / "u_scaler.pkl")
        joblib.dump(self.x_scaler, run_dir / "x_scaler.pkl")