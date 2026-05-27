import joblib

from .processor import Processor
from .loader import Loader

class Builder:
    def __init__(self, dataset_paths, params: dict, drone_dim: int):
        self.loader = Loader(
            dataset_paths,
            drone_dim,
            params["train"],
            params["val_datasets"][0],
            params["val_datasets"][1],
            params["downsample_factor"],
            True,
            )
        raw_data = self.loader.load_raw_sensor_data()
        batch_size = params["batch_size"]

        self.processor = Processor(
            batch_size,
            params["train"],
            params["val_datasets"][0],
            params["val_datasets"][1],
            raw_data,
            params["scaler"],
            params["delay"],
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