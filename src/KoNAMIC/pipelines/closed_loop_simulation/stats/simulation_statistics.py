from dataclasses import dataclass

@dataclass
class SimulationStatistics:
    """Regroupe les indicateurs de performance d'une simulation."""
    error_x_pos: float
    error_z: float
    terminal_error: float
    sqp_iter: float
    has_diverged: bool
    overshoot_1: float
    overshoot_2: float
    settling_time_1: float
    settling_time_2: float

    def as_dict(self) -> dict:
        """Conversion pratique pour stockage dans metrics."""
        return {
            "error_x": self.error_x_pos,
            "error_z": self.error_z,
            "terminal_error": self.terminal_error,
            "sqp_iter": self.sqp_iter,
            "divergence_flag": self.has_diverged,
            "overshoot_rate_x_1": self.overshoot_1,
            "overshoot_rate_x_2": self.overshoot_2,
            "settling_time_x_1": self.settling_time_1,
            "settling_time_x_2": self.settling_time_2,
        }

    def short_summary(self) -> str:
        """Retourne une ligne de résumé texte lisible."""
        div = "❌ DIVERGED" if self.has_diverged else "✅ OK"
        summary = (
            f"Error on x: {self.error_x_pos:.2e}, "
            f"Error on z: {self.error_z:.2e}, "
            f"term err: {self.terminal_error:.2e}, "
            f"SQP: {self.sqp_iter:.1f}, "
            f"OS: ({self.overshoot_1:.1f}%, {self.overshoot_2:.1f}%), "
            f"Ts: ({self.settling_time_1:.2f}s, {self.settling_time_2:.2f}s), "
            f"{div}"
                )
        return summary
