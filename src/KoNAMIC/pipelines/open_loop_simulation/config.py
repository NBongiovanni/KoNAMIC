from dataclasses import dataclass
from typing import Any

from KoNAMIC.config.config_utils import require_keys


@dataclass(frozen=True)
class OpenLoopEvalConfig:
    system_name: str
    modality: str
    plot_every: int
    num_visualized_rollouts: int
    num_steps_simulation: int
    save_plots: bool

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "OpenLoopEvalConfig":
        require_keys(
            cfg,
            [
                "system_name",
                "modality",
                "plot_every",
                "num_visualized_rollouts",
                "num_steps_simulation",
                "save_plots",
            ],
            "open_loop_eval",
        )
        return cls(
            system_name=str(cfg["system_name"]),
            modality=str(cfg["modality"]),
            plot_every=int(cfg["plot_every"]),
            num_visualized_rollouts=int(cfg["num_visualized_rollouts"]),
            num_steps_simulation=int(cfg["num_steps_simulation"]),
            save_plots=bool(cfg["save_plots"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_name": self.system_name,
            "modality": self.modality,
            "plot_every": self.plot_every,
            "num_visualized_rollouts": self.num_visualized_rollouts,
            "num_steps_simulation": self.num_steps_simulation,
            "save_plots": self.save_plots,
        }
