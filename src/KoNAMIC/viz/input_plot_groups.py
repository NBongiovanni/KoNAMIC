from __future__ import annotations

from KoNAMIC.core.systems import SystemSpec
from KoNAMIC.viz.primitives_single import InputPlotGroup


def build_input_plot_groups(
    system_spec: SystemSpec,
    *,
    group_inputs: bool = True,
) -> list[InputPlotGroup]:
    groups = system_spec.get_input_plot_groups(group_inputs=group_inputs)
    labels = system_spec.get_u_labels()

    input_plot_groups: list[InputPlotGroup] = []
    for group in groups:
        indices = tuple(int(idx) for idx in group["indices"])
        legend_labels = None
        if len(indices) > 1:
            legend_labels = tuple(labels[idx] for idx in indices)

        input_plot_groups.append(
            InputPlotGroup(
                indices=indices,
                ylabel=str(group["label"]),
                legend_labels=legend_labels,
            )
        )

    return input_plot_groups
