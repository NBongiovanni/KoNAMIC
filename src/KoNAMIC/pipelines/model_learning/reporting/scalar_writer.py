from __future__ import annotations

from .backends import MetricsBackend


class ScalarWriter:
    def __init__(self, backend: MetricsBackend) -> None:
        self.backend = backend

    def scalar(self, tag_base: str, name: str, quantity, epoch: int) -> None:
        self.backend.log_scalar(tag_base + name, quantity, epoch)

    def scalars(self, epoch: int, tag_base: str, scalars: dict[str, float]) -> None:
        for name, value in scalars.items():
            self.scalar(tag_base, name, value, epoch)

    def optimizer_state(self, tag_base: str, optimizer, epoch: int) -> None:
        for i, param_group in enumerate(optimizer.param_groups):
            group_name = param_group.get("name", f"group_{i}")
            self.scalar(
                tag_base,
                f"lr/{group_name}",
                param_group["lr"],
                epoch,
            )
