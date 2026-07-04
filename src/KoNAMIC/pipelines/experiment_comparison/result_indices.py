from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .sources import ComparisonSource


def resolve_common_result_indices(
    *,
    sources: Sequence[ComparisonSource],
    directory_prefix: str,
    result_filename: str,
    source_kind_label: str,
    empty_intersection_message: str,
) -> list[int]:
    if not sources:
        raise ValueError("At least one comparison source is required.")

    index_sets = []

    for source in sources:
        if not source.results_dir.exists():
            raise FileNotFoundError(
                f"Missing results directory for {source_kind_label} "
                f"'{source.label}': {source.results_dir}"
            )

        indices = {
            int(path.name.removeprefix(directory_prefix))
            for path in source.results_dir.iterdir()
            if _is_indexed_result_dir(
                path=path,
                directory_prefix=directory_prefix,
                result_filename=result_filename,
            )
        }

        if not indices:
            raise FileNotFoundError(
                f"No {directory_prefix}*/{result_filename} files found for "
                f"'{source.label}' in {source.results_dir}"
            )

        index_sets.append(indices)

    common_indices = set.intersection(*index_sets)
    if not common_indices:
        raise FileNotFoundError(empty_intersection_message)

    return sorted(common_indices)


def build_indexed_result_path(
    *,
    source: ComparisonSource,
    directory_prefix: str,
    result_index: int,
    result_filename: str,
) -> Path:
    return source.results_dir / f"{directory_prefix}{result_index}" / result_filename


def _is_indexed_result_dir(
    *,
    path: Path,
    directory_prefix: str,
    result_filename: str,
) -> bool:
    index_text = path.name.removeprefix(directory_prefix)
    return (
        path.is_dir()
        and path.name.startswith(directory_prefix)
        and index_text.isdigit()
        and (path / result_filename).exists()
    )
