from __future__ import annotations

from collections.abc import Sequence


def select_one_traj_per_profile(
    profiles: Sequence[str],
) -> tuple[int, ...]:
    """
    Return the first trajectory index observed for each profile.

    The profile assignment itself must come from ScenarioGenerator.
    This function only selects representative indices for diagnostics.
    """
    traj_indices_by_profile: dict[str, int] = {}

    for traj_idx, profile in enumerate(profiles):
        if profile not in traj_indices_by_profile:
            traj_indices_by_profile[profile] = traj_idx

    return tuple(traj_indices_by_profile.values())