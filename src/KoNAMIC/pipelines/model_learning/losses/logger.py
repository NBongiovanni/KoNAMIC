from __future__ import annotations

from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


VISION_TAGS = {
    "y_pred": "01_y_pred",
    "y_rec": "02_y_rec",
    "z": "03_z",
    "c": "04_c",
    "horizontal": "05_horizontal",
    "vertical": "06_vertical",
    "angle": "07_angle",
    "iou": "08_iou",
}

SENSOR_TAGS = {
    "pred_x": "01_x_pred",
    "rec": "02_x_rec",
    "pred_z": "03_z",
    "pred_position": "04_position",
}


class TrainingLogger:
    def __init__(
        self,
        modality: str,
        writer: SummaryWriter,
        num_val_datasets: int,
    ) -> None:
        self.modality = modality
        self.writer = writer
        self.num_val_datasets = num_val_datasets
        self.tags = self._get_tags(modality)

    def log_losses(
        self,
        epoch: int,
        phase_losses: dict,
        closed_loop_metrics=None,
    ) -> None:
        self._log_losses_terminal(
            epoch=epoch,
            phase_losses=phase_losses,
            closed_loop_metrics=closed_loop_metrics,
        )
        self._log_losses_tensorboard(
            epoch=epoch,
            phase_losses=phase_losses,
        )

    def log_training_state(
            self,
            epoch: int,
            optimizer: Optimizer,
            total_norm: float,
            *,
            base: dict | None = None,
            effective_weight=None,
            phase_index: int | None = None,
    ) -> None:
        if self.modality == "sensor":
            self._log_training_state_sensor(
                epoch=epoch,
                optimizer=optimizer,
                total_norm=total_norm,
            )
        elif self.modality == "vision":
            self._log_training_state_vision(
                epoch=epoch,
                optimizer=optimizer,
                base=base,
                effective_weight=effective_weight,
                total_norm=total_norm,
                phase_index=phase_index,
            )
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

    def _log_training_state_sensor(
        self,
        epoch: int,
        optimizer: Optimizer,
        total_norm: float,
    ) -> None:
        tag = "08_training_state/"

        for i, pg in enumerate(optimizer.param_groups):
            group_name = pg.get("name", f"group_{i}")
            self.writer.add_scalar(tag + f"lr/{group_name}", pg["lr"], epoch)

        self.writer.add_scalar(tag + "grad_norm", total_norm, epoch)

    def _log_training_state_vision(
        self,
        epoch: int,
        optimizer: Optimizer,
        total_norm: float,
        base: dict | None = None,
        effective_weight=None,
        phase_index: int | None = None,
    ) -> None:
        tag = "08_training_state/"

        self._log_on_tb(tag, "c_effective", effective_weight(base["c"], "c"), epoch)
        self._log_on_tb(tag, "a_effective", effective_weight(base["a"], "a"), epoch)
        self._log_on_tb(tag, "phase_index", phase_index, epoch)

        for i, pg in enumerate(optimizer.param_groups):
            group_name = pg.get("name", f"group_{i}")
            self._log_on_tb(tag, f"lr/{group_name}", pg["lr"], epoch)

        self._log_on_tb(tag, "grad_norm", total_norm, epoch)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()

    def _log_losses_terminal(
        self,
        epoch: int,
        phase_losses: dict,
        closed_loop_metrics=None,
    ) -> None:
        phase_names = self._phase_names()

        print(f"Epoch: {epoch}")
        for phase in phase_names:
            global_loss, sub = phase_losses[phase]
            lines = [f"{phase} loss: {global_loss:.2e}", "\t  pred :"]

            for key, tag in self.tags.items():
                val = sub[key]
                lines.append(f"\t    {tag:<9}: {val:.2e}")

            print("\n".join(lines))

        print("")

        if closed_loop_metrics is not None:
            print("closed-loop:")
            print(f"\tposition RMSE: {closed_loop_metrics.position_rmse:.2e}")
            print(f"\tx RMSE       : {closed_loop_metrics.x_rmse:.2e}")
            print(f"\tz RMSE       : {closed_loop_metrics.z_rmse:.2e}")
            print(f"\tu RMS        : {closed_loop_metrics.u_rms:.2e}")

    def _log_losses_tensorboard(
        self,
        epoch: int,
        phase_losses: dict,
    ) -> None:
        phase_idx = 2

        for phase in self._phase_names():
            global_loss, sub = phase_losses[phase]

            self.writer.add_scalar(f"01_total/{phase}", global_loss, epoch)

            tag_base = f"0{phase_idx}_{phase}/"
            for key, tag in self.tags.items():
                self.writer.add_scalar(tag_base + tag, sub[key], epoch)

            phase_idx += 1

    def _log_on_tb(
        self,
        tag: str,
        name: str,
        quantity,
        epoch: int,
    ) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag + name, quantity, epoch)

    def _phase_names(self) -> list[str]:
        return ["train"] + [
            f"val_{i}" for i in range(1, self.num_val_datasets + 1)
        ]

    @staticmethod
    def _get_tags(modality: str) -> dict[str, str]:
        if modality == "vision":
            return VISION_TAGS
        if modality == "sensor":
            return SENSOR_TAGS
        raise ValueError(f"Unknown modality: {modality}")