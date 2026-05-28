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
