import joblib

from .sensor_processor import SensorProcessor
from .loader import SensorLoader, SensorLoadSpec

class SensorBuilder:
    def __init__(self, dataset_paths, params, drone_dim: int):

        split_specs = {
            "train": SensorLoadSpec(num_steps_loaded=None),
            "val_1": SensorLoadSpec(num_steps_loaded=None),
            "val_2": SensorLoadSpec(num_steps_loaded=None),
        }

        self.loader = SensorLoader(
            dataset_paths=dataset_paths,
            drone_dim=drone_dim,
            split_specs=split_specs,
            downsample_factor=1,
        )
        raw_data = self.loader.load_raw_sensor_data()
        batch_size = params["dataloader"]["batch_size"]

        self.processor = SensorProcessor(
            batch_size,
            params["train"],
            params["val_datasets"][0],
            params["val_datasets"][1],
            raw_data,
            params["scaler"],
            params["postprocessing"]["delay"],
        )
        self.processor.generate_u_scaler()
        self.processor.generate_x_scaler()

        self.processor.process_datasets()
        self.u_scaler = self.processor.u_scaler
        self.x_scaler = self.processor.x_scaler

        self.processor.build_data_loader()
        self.data_loader = self.processor.data_loader

    @property
    def processed(self):
        return self.processor.processed_datasets

    def save_scalers(self, run_dir) -> None:
        (run_dir / "scalers").mkdir(exist_ok=True)
        joblib.dump(self.u_scaler, run_dir / "u_scaler.pkl")
        joblib.dump(self.x_scaler, run_dir / "x_scaler.pkl")