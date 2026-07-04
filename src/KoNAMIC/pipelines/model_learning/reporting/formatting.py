NAME_WIDTH = 24
VALUE_WIDTH = 10


def format_metric_line(
    name: str,
    value: float,
    *,
    indent: str = "\t  ",
) -> str:
    return f"{indent}{name:<{NAME_WIDTH}}: {value:>{VALUE_WIDTH}.2e}"


def format_metric_columns(
    metrics: dict[str, float],
    *,
    indent: str = "\t  ",
    n_cols: int = 4,
    name_width: int = 12,
    value_width: int = VALUE_WIDTH,
    col_gap: str = "    ",
) -> list[str]:
    items = list(metrics.items())

    if not items:
        return []

    n_rows = (len(items) + n_cols - 1) // n_cols
    lines = []

    for row_idx in range(n_rows):
        row_parts = []

        for col_idx in range(n_cols):
            item_idx = row_idx + col_idx * n_rows

            if item_idx >= len(items):
                continue

            name, value = items[item_idx]
            row_parts.append(
                f"{name:<{name_width}}: {value:>{value_width}.2e}"
            )

        lines.append(indent + col_gap.join(row_parts))

    return lines
