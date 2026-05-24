from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.models import SensorKoopModel
from KoNAMIC.core.simulation import compute_closed_loop_metrics

from ..training.config import PredictionHorizon
from ..losses.classes import EpochEvalResult, OpenLoopLosses
from .closed_loop_eval import ClosedLoopEval, ClosedLoopTrajectory

SubLosses: TypeAlias = dict[str, float]


class ModelEvaluator:
    """
    Gère l'évaluation d'un modèle pendant l'entraînement.

    Pour l'instant :
      - validation open-loop sur val_1 et val_2
    Plus tard :
      - évaluation closed-loop périodique
    TODO: replace SensorKoopModel by BaseKoopModel
    """

    def __init__(
        self,
        modality: str,
        model_params: dict,
        control_params: dict,
        drone: DroneSpec,
        koop_model: SensorKoopModel,
        data_loaders: dict,
        device: torch.device,
        prediction_horizon: PredictionHorizon,
        reduce_sub_losses_fn,
        forward_and_loss_fn,
        x_scaler: StandardScaler,
        u_scaler: StandardScaler,
        closed_loop_every: int,
    ) -> None:

        self.modality = modality
        self.model_params = model_params
        self.control_params = control_params
        self.drone = drone
        self.koop_model = koop_model
        self.data_loader = data_loaders
        self.device = device

        self.num_steps_val_1 = prediction_horizon.val[0]
        self.num_steps_val_2 = prediction_horizon.val[1]

        self.reduce_sub_losses = reduce_sub_losses_fn
        self.forward_and_loss = forward_and_loss_fn

        self.x_scaler = x_scaler
        self.u_scaler = u_scaler

        self.closed_loop_every = closed_loop_every

    @torch.no_grad()
    def eval_open_loop(self, phase: str, num_steps: int) -> OpenLoopLosses:
        self.koop_model.eval()

        full_loss_arr: list[float] = []
        sub_losses_arr: list[SubLosses] = []

        for batch in tqdm(self.data_loader[phase]):
            full_loss, sub_losses = self.forward_and_loss(batch, num_steps)
            full_loss_arr.append(float(full_loss.cpu().item()))
            sub_losses_arr.append(sub_losses)

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = self.reduce_sub_losses(sub_losses_arr)
        return OpenLoopLosses(full_loss=full_mean, sub_losses=sub_mean)

    def should_run_closed_loop(self, epoch: int) -> bool:
        if self.closed_loop_every is None:
            return False
        if self.closed_loop_every <= 0:
            return False
        return epoch % self.closed_loop_every == 0

    def eval_closed_loop(self) -> list[ClosedLoopTrajectory]:
        closed_loop_evaluator = ClosedLoopEval(
            self.modality,
            self.model_params,
            self.control_params,
            self.koop_model,
            self.drone,
            self.u_scaler,
            self.x_scaler,
        )
        list_results = closed_loop_evaluator.run_simulation()
        return list_results


    def evaluate_epoch(self, epoch: int) -> EpochEvalResult:
        val_1_result = self.eval_open_loop("val_1", self.num_steps_val_1)
        val_2_result = self.eval_open_loop("val_2", self.num_steps_val_2)
        if self.should_run_closed_loop(epoch):
            closed_loop_result = self.eval_closed_loop()
            closed_loop_metrics = compute_closed_loop_metrics(closed_loop_result)
        else:
            closed_loop_result = None
            closed_loop_metrics = None

        return EpochEvalResult(
            val_1=val_1_result,
            val_2=val_2_result,
            closed_loop_trajectories=closed_loop_result,
            closed_loop_metrics=closed_loop_metrics,
        )