from __future__ import annotations

from collections.abc import Callable, Sequence
from logging import Logger
from pathlib import Path
from typing import Protocol, TypeVar

from .result_indices import build_indexed_result_path
from .sources import ComparisonSource


ResultT = TypeVar("ResultT")


class ComparisonVisualizer(Protocol[ResultT]):
    def visualize(self, results: Sequence[ResultT]) -> None:
        ...


LoadResultCallback = Callable[[ComparisonSource, Path], ResultT]
MakeVisualizerCallback = Callable[[int, Path, Sequence[ResultT]], ComparisonVisualizer[ResultT]]


def run_indexed_comparison(
    *,
    sources: Sequence[ComparisonSource],
    indices: Sequence[int],
    output_dir: Path,
    directory_prefix: str,
    result_filename: str,
    item_label: str,
    logger: Logger,
    load_result: LoadResultCallback[ResultT],
    make_visualizer: MakeVisualizerCallback[ResultT],
) -> None:
    if not indices:
        raise ValueError("At least one comparison index is required.")

    for index in indices:
        logger.info(
            "Processing %s %s%d / %s%d",
            item_label,
            directory_prefix,
            index,
            directory_prefix,
            indices[-1],
        )

        item_output_dir = output_dir / f"{directory_prefix}{index}"
        item_output_dir.mkdir(parents=True, exist_ok=True)

        results: list[ResultT] = []
        for source in sources:
            result_path = build_indexed_result_path(
                source=source,
                directory_prefix=directory_prefix,
                result_index=index,
                result_filename=result_filename,
            )

            if not result_path.exists():
                raise FileNotFoundError(
                    f"Missing results file for '{source.label}': {result_path}"
                )

            logger.info("Loading results for '%s' from %s", source.label, result_path)
            results.append(load_result(source, result_path))

        visualizer = make_visualizer(index, item_output_dir, results)
        visualizer.visualize(results)
