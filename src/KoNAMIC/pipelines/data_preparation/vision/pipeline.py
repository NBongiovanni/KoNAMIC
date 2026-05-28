from .processor import VisionProcessor


def iter_sample_configs(dataset_params: dict) -> list[dict]:
    return [
        {
            "split": "train",
            "name": "train",
            **dataset_params["train"],
        },
        *dataset_params["val_datasets"],
    ]



def prepare_vision_memmap(
    num_traj: int,
    dataset_params,
    dataset_stamp: str,
) -> None:

    for sample_cfg in iter_sample_configs(dataset_params):
        phase = sample_cfg["split"]

        num_steps_loaded = sample_cfg["num_steps_loaded"]
        num_steps_pred = sample_cfg["num_steps_pred"]

        print(
            f"[INFO] Preparing vision memmap for {phase}: "
            f"num_traj={num_traj}, "
            f"num_steps_loaded={num_steps_loaded}, "
            f"num_steps_pred={num_steps_pred}"
        )

        vision_processor = VisionProcessor(
            dataset_params,
            phase,
            dataset_stamp,
            num_steps_loaded,
        )

        vision_processor.pipeline(
            num_traj,
            dataset_params["resolution"],
            num_steps_pred,
        )