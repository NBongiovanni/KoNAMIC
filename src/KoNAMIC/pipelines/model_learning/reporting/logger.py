from __future__ import annotations

from collections.abc import Callable
from typing import Any

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.pipelines.model_learning.losses.classes import OpenLoopLosses
from .backends import MetricsBackend
from .formatting import format_metric_columns, format_metric_line
from .scalar_writer import ScalarWriter
from .tags import (
    build_closed_loop_summary_tags,
    build_state_error_tags,
    get_error_state_labels,
    get_tags,
)

# TODO: make it work with visual data


class TrainingLogger:
    def __init__(
        self,
        modality: str,
        backend: MetricsBackend,
        num_val_datasets: int,
        system_spec: SystemSpec,
    ) -> None:

        self.modality = modality
        self.backend = backend
        self.scalar_writer = ScalarWriter(backend)
        self.num_val_datasets = num_val_datasets
        self.system_spec = system_spec

        self.tags = get_tags(modality)
        self.error_state_labels = get_error_state_labels(modality, system_spec)

    def log_losses(
        self,
        epoch: int,
        phase_losses: dict[str, OpenLoopLosses],
        closed_loop_metrics: Any | None = None,
    ) -> None:
        self._log_open_loop_losses_terminal(epoch, phase_losses)
        self._log_open_loop_losses_backend(epoch, phase_losses)

        if closed_loop_metrics is not None:
            self._log_closed_loop_terminal(closed_loop_metrics)
            self._log_closed_loop_backend(epoch, closed_loop_metrics)

    def log_training_state(
        self,
        epoch: int,
        optimizer,
        total_norm: float,
        *,
        base: dict | None = None,
        effective_weight: Callable | None = None,
        phase_index: int | None = None,
    ) -> None:
        tag_base = "08_training_state/"

        if self.modality == "vision":
            if base is None or effective_weight is None:
                raise ValueError(
                    "base and effective_weight must be provided for vision logging."
                )

            self.scalar_writer.scalar(
                tag_base,
                "c_effective",
                effective_weight(base["c"], "c"),
                epoch,
            )
            self.scalar_writer.scalar(
                tag_base,
                "a_effective",
                effective_weight(base["a"], "a"),
                epoch,
            )

            if phase_index is not None:
                self.scalar_writer.scalar(tag_base, "phase_index", phase_index, epoch)

        elif self.modality != "sensor":
            raise ValueError(f"Unknown modality: {self.modality}")

        self.scalar_writer.optimizer_state(tag_base, optimizer, epoch)
        self.scalar_writer.scalar(tag_base, "grad_norm", total_norm, epoch)

    def _log_open_loop_losses_terminal(
        self,
        epoch: int,
        phase_losses: dict[str, OpenLoopLosses],
    ) -> None:
        print(f"Epoch: {epoch}")

        for phase in self._phase_names():
            losses = phase_losses[phase]
            sub = losses.sub_losses
            lines = [f"{phase} loss: {losses.full_loss:.2e}"]

            for _, tag, value in self._iter_loss_tags(sub):
                lines.append(format_metric_line(tag, value))

            state_rmse = sub.get("state_rmse")
            if state_rmse is not None:
                lines.append("\t  state rmse:")
                lines.extend(
                    format_metric_columns(
                        state_rmse,
                        indent="\t  ",
                        n_cols=4,
                    )
                )

            print("\n".join(lines))
        print("")

    def _log_open_loop_losses_backend(
        self,
        epoch: int,
        phase_losses: dict[str, OpenLoopLosses],
    ) -> None:
        for phase_idx, phase in enumerate(self._phase_names(), start=2):
            losses = phase_losses[phase]
            sub = losses.sub_losses

            self.scalar_writer.scalar("01_total/", phase, losses.full_loss, epoch)

            tag_base = f"0{phase_idx}_{phase}/"
            for _, tag, value in self._iter_loss_tags(sub):
                self.scalar_writer.scalar(tag_base, tag, value, epoch)

            state_rmse = sub.get("state_rmse")
            if state_rmse is not None:
                prefix = tag_base + self.tags["open_loop"]["state_rmse"]

                state_tags = build_state_error_tags(
                    prefix=prefix,
                    state_labels=list(state_rmse.keys()),
                )

                for label, value in state_rmse.items():
                    self.backend.log_scalar(state_tags[label], value, epoch)

    def _log_closed_loop_terminal(self, metrics: Any) -> None:
        print("closed-loop:")

        for attr, _, value in self._iter_closed_loop_metrics(metrics):
            label = attr.replace("_", " ")
            print(
                format_metric_line(
                    label,
                    value,
                    indent="\t",
                )
            )
        state_rmse = getattr(metrics, "state_rmse", None)

        if state_rmse is not None:
            print("\tstate rmse:")

            for line in format_metric_columns(
                state_rmse,
                indent="\t  ",
                n_cols=4,
            ):
                print(line)

    def _log_closed_loop_backend(self, epoch: int, metrics: Any) -> None:
        for _, tag, value in self._iter_closed_loop_metrics(metrics):
            self.backend.log_scalar(tag, value, epoch)

        state_rmse = getattr(metrics, "state_rmse", None)

        if state_rmse is not None:
            prefix = self.tags["closed_loop"]["state_rmse"]

            state_tags = build_state_error_tags(
                prefix=prefix,
                state_labels=list(state_rmse.keys()),
            )

            for label, value in state_rmse.items():
                self.backend.log_scalar(state_tags[label], value, epoch)

    def _iter_closed_loop_metrics(self, metrics: Any):
        metric_tags = build_closed_loop_summary_tags(self.tags)

        for attr, tag in metric_tags.items():
            if hasattr(metrics, attr):
                yield attr, tag, getattr(metrics, attr)

    def _phase_names(self) -> list[str]:
        return ["train"] + [
            f"val_{i}" for i in range(1, self.num_val_datasets + 1)
        ]

    def log_scalars(
        self,
        *,
        epoch: int,
        tag_base: str,
        scalars: dict[str, float],
    ) -> None:
        self.scalar_writer.scalars(
            epoch=epoch,
            tag_base=tag_base,
            scalars=scalars,
        )

    def close(self) -> None:
        self.backend.close()

    def _iter_loss_tags(self, sub: dict):
        for section in ["reconstruction", "open_loop"]:
            for key, tag in self.tags[section].items():
                if key in sub and isinstance(sub[key], float):
                    yield key, tag, sub[key]
