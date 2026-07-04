from __future__ import annotations

from typing import Any, Protocol

from KoNAMIC.config.config_utils import make_serializable


class MetricsBackend(Protocol):
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        ...

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        ...

    def log_config(self, config: dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class TensorBoardBackend:
    def __init__(self, writer) -> None:
        self.writer = writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)

    def log_config(self, config: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class WandbBackend:
    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        entity: str | None = None,
        mode: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=make_serializable(config or {}),
            entity=entity,
            mode=mode,
            tags=tags,
        )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.wandb.log({tag: value}, step=step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        self.wandb.log(scalars, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self.run.config.update(make_serializable(config), allow_val_change=True)

    def close(self) -> None:
        self.run.finish()


class MultiBackend:
    def __init__(self, backends: list[MetricsBackend]) -> None:
        self.backends = backends

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for backend in self.backends:
            backend.log_scalar(tag, value, step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for backend in self.backends:
            backend.log_scalars(scalars, step)

    def log_config(self, config: dict[str, Any]) -> None:
        for backend in self.backends:
            backend.log_config(config)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()


class NullBackend:
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        return None

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        return None

    def log_config(self, config: dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


def build_metrics_backend(
    *,
    writer,
    logging_config,
    run_name: str | None = None,
    run_config: dict[str, Any] | None = None,
) -> MetricsBackend:
    if logging_config is None:
        return TensorBoardBackend(writer)

    backend = logging_config.backend
    if backend == "tensorboard":
        return TensorBoardBackend(writer)

    if backend == "none":
        return NullBackend()

    if backend in {"wandb", "both"}:
        wandb_cfg = logging_config.wandb
        if wandb_cfg is None:
            raise ValueError(
                "logging_config.wandb must be set when using W&B logging."
            )
        wandb_backend = WandbBackend(
            project=wandb_cfg.project,
            run_name=run_name,
            config=run_config,
            entity=wandb_cfg.entity,
            mode=wandb_cfg.mode,
            tags=wandb_cfg.tags,
        )
        if backend == "wandb":
            return wandb_backend
        return MultiBackend([TensorBoardBackend(writer), wandb_backend])

    raise ValueError(f"Unknown logging backend: {backend!r}")
