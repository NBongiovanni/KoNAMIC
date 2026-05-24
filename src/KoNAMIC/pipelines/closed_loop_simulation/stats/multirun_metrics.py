from dataclasses import dataclass, field
import numpy as np
from KoNAMIC.pipelines.closed_loop_simulation.stats.simulation_statistics import SimulationStatistics

@dataclass
class MultiRunMetrics:
    mse_x: list[float] = field(default_factory=list)
    mse_z: list[float] = field(default_factory=list)
    terminal_error: list[float] = field(default_factory=list)
    sqp_iters: list[float] = field(default_factory=list)
    divergence_flags: list[bool] = field(default_factory=list)
    overshoot_rate_x_1: list[float] = field(default_factory=list)
    overshoot_rate_x_2: list[float] = field(default_factory=list)
    settling_time_x_1: list[float] = field(default_factory=list)
    settling_time_x_2: list[float] = field(default_factory=list)

    def add(self, s: SimulationStatistics) -> None:
        self.mse_x.append(s.error_x_pos)
        self.mse_z.append(s.error_z)
        self.terminal_error.append(s.terminal_error)
        self.sqp_iters.append(s.sqp_iter)
        self.divergence_flags.append(s.has_diverged)
        self.overshoot_rate_x_1.append(s.overshoot_1)
        self.overshoot_rate_x_2.append(s.overshoot_2)
        self.settling_time_x_1.append(s.settling_time_1)
        self.settling_time_x_2.append(s.settling_time_2)

    # Pour compatibilité immédiate avec display_multi_run_statistics
    def as_lists(self) -> dict:
        return {
            "mse_x": self.mse_x,
            "mse_z": self.mse_z,
            "terminal_error": self.terminal_error,
            "sqp_iters": self.sqp_iters,
            "divergence_flags": self.divergence_flags,
            "overshoot_rate_x_1": self.overshoot_rate_x_1,
            "overshoot_rate_x_2": self.overshoot_rate_x_2,
            "settling_time_x_1": self.settling_time_x_1,
            "settling_time_x_2": self.settling_time_x_2,
        }

    # Optionnel : petits agrégats utiles
    def summary(self) -> dict:
        nmean = lambda v: float(np.nanmean(np.asarray(v, dtype=float))) if len(v) else float("nan")
        return {
            "RMSE_pos": float(np.sqrt(nmean(self.mse_x))),
            "Ts_mean": nmean(self.settling_time_x_1 + self.settling_time_x_2),
            "OS_mean": nmean(self.overshoot_rate_x_1 + self.overshoot_rate_x_2),
            "Terminal": nmean(self.terminal_error),
            "SQP_iter": nmean(self.sqp_iters),
            "Divergence": nmean(self.divergence_flags),
        }

    def _print_statistics(self, arr: np.ndarray) -> None:
        print(f"  Mean:      {np.nanmean(arr):.4e}")
        print(f"  Std:       {np.nanstd(arr):.4e}")

    def display(self, num_simulations: int) -> None:
        num_diverged = sum(self.divergence_flags)
        success_rate = 100 * (1 - num_diverged / max(1, num_simulations))

        print(f"📊 Performance Metrics")
        print(f"{'─' * 70}")
        print(f"  Total simulations:     {num_simulations}")
        print(f"  Successful:            {num_simulations - num_diverged} ({success_rate:.1f}%)")
        print(f"  Diverged:              {num_diverged}\n")

        print(f"📈 Mean errors on Physical State (x, y positions)")
        print(f"{'─' * 70}")
        self._print_statistics(self.mse_x); print()

        print(f"📈 Error on Latent State (z)")
        print(f"{'─' * 70}")
        self._print_statistics(np.asarray(self.mse_z, dtype=float)); print()

        print(f"🎯 Terminal Error (L2 norm at final step) [m]")
        print(f"{'─' * 70}")
        self._print_statistics(np.asarray(self.terminal_error, dtype=float)); print()

        print(f"⚙️  Mean SQP iterations")
        print(f"{'─' * 70}")
        self._print_statistics(np.asarray(self.sqp_iters, dtype=float))
        if num_diverged > 0:
            print(f"⚠️  Diverged simulations: "
                  f"{[i + 1 for i, d in enumerate(self.divergence_flags) if d]}")
        print()

        print(f"📌 Overshoot (%) — coordonnées 1 et 2")
        print(f"{'─' * 70}")
        self._print_statistics(np.asarray(self.overshoot_rate_x_1, dtype=float))
        self._print_statistics(np.asarray(self.overshoot_rate_x_2, dtype=float)); print()

        print(f"📌 Settling times (s) — coordonnées 1 et 2")
        print(f"{'─' * 70}")
        self._print_statistics(np.asarray(self.settling_time_x_1, dtype=float))
        self._print_statistics(np.asarray(self.settling_time_x_2, dtype=float))
