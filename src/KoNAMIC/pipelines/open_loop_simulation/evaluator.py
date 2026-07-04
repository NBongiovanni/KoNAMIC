from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from KoNAMIC import config
from KoNAMIC.pipelines.model_learning.losses.classes import OpenLoopLosses
from KoNAMIC.pipelines.model_learning.losses.utils import mean_sub_losses
from KoNAMIC.core.models.model_config import ModelConfig
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.paths import RunPaths
from KoNAMIC.pipelines.model_learning.training.forward_loss_computer import (
    SensorForwardLossComputer,
    VisionForwardLossComputer,
)

from .config import OpenLoopEvalConfig
from .post_process import make_sensor_rollout_output, make_vision_rollout_output
from .rollout_extractors import get_rollout_extractor_for_modality
from .viz.render_open_loop_rollouts import RenderOpenLoopConfig, render_open_loop_rollouts
SubLosses: TypeAlias = dict[str, float]


class OpenLoopEvaluator:
    """
    Évalue un modèle Koopman en open loop sur un dataloader donné.
    """

    def __init__(
        self,
        *,
        forward_loss_computer: SensorForwardLossComputer | VisionForwardLossComputer,
        data_loaders: Mapping[str, DataLoader],
        scalers: DatasetScalers,
        system_spec: SystemSpec,
        run_paths: RunPaths,
        model_config: ModelConfig,
        open_loop_eval_config: OpenLoopEvalConfig,
        modality: config.Modality,
    ) -> None:
        self.forward_loss_computer = forward_loss_computer
        self.data_loaders = data_loaders
        self.koop_model = forward_loss_computer.koop_model
        self.scalers = scalers
        self.system_spec = system_spec
        self.run_paths = run_paths
        self.model_config = model_config
        self.open_loop_eval_config = open_loop_eval_config
        self.modality = modality

    @torch.no_grad()
    def evaluate(self, *, phase: str, num_steps: int) -> OpenLoopLosses:
        self.koop_model.eval()

        full_loss_arr: list[float] = []
        sub_losses_arr: list[SubLosses] = []

        for batch in tqdm(self.data_loaders[phase]):
            full_loss, sub_losses = self.forward_loss_computer.compute(
                batch=batch,
                num_steps=num_steps,
            )
            full_loss_arr.append(float(full_loss.cpu().item()))
            sub_losses_arr.append(sub_losses)

        full_mean = float(np.mean(full_loss_arr)) if full_loss_arr else float("nan")
        sub_mean = self.reduce_sub_losses(sub_losses_arr)

        return OpenLoopLosses(
            full_loss=full_mean,
            sub_losses=sub_mean,
        )

    def render_sample_rollouts(
        self,
        *,
        epoch: int,
        phase: str,
        num_steps: int,
        only_position: bool = False,
    ) -> None:
        self.koop_model.eval()
        device = next(self.koop_model.parameters()).device

        if self.modality is config.Modality.SENSOR:
            output = make_sensor_rollout_output(
                koop_model=self.koop_model,
                dataloader=self.data_loaders[phase],
                x_scaler=self.scalers.x,
                u_scaler=self.scalers.u,
                system_spec=self.system_spec,
                num_steps=num_steps,
                device=device,
            )
            batch_size = output.state_gt_scaled.shape[0]
            render_only_position = only_position
        elif self.modality is config.Modality.VISION:
            output = make_vision_rollout_output(
                koop_model=self.koop_model,
                dataloader=self.data_loaders[phase],
                u_scaler=self.scalers.u,
                num_steps=num_steps,
                device=device,
            )
            batch_size = output.g_t.state.shape[0]
            render_only_position = True
        else:
            raise ValueError(f"Unknown modality: {self.modality.key}")

        num_rollouts = min(
            self.open_loop_eval_config.num_visualized_rollouts,
            batch_size,
        )
        if num_rollouts <= 0:
            return

        eval_dir = self.run_paths.training_eval_dir("open_loop", epoch) / phase
        render_config = RenderOpenLoopConfig(
            modality=self.modality,
            system_dim=self.system_spec.system_dim,
            dt=self.model_config.dt,
            phase=phase,
            epoch=epoch,
            num_rollouts=num_rollouts,
            num_columns_states=2,
            num_columns_inputs=2,
            only_position=render_only_position,
            render_images=False,
            snapshots=False,
            num_steps=num_steps,
            label=f"epoch_{epoch:04d}",
        )

        render_open_loop_rollouts(
            config=render_config,
            output=output,
            eval_dir=eval_dir,
            extract_one_rollout=get_rollout_extractor_for_modality(self.modality),
        )

    @staticmethod
    def reduce_sub_losses(sub_losses_arr: list[SubLosses]) -> SubLosses:
        return mean_sub_losses(sub_losses_arr)
