from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from KoNAMIC.paths import build_dataset_paths
from .config import VisionPreparationConfig


class VisionProcessor:
    def __init__(
            self,
            params: VisionPreparationConfig,
            phase: str,
            dataset_stamp: str,
            num_steps_loaded: int,
    ) -> None:
        super().__init__()

        self.params = params
        self.phase = phase
        self.dataset_stamp = dataset_stamp
        self.num_steps_loaded = num_steps_loaded
        self.system_name = params.system_name
        self.drone_dim = params.drone_dim

        self.dataset_paths = build_dataset_paths(self.system_name, self.dataset_stamp)
        self.dataset_states = None
        self.dataset_inputs = None
        self.dataset_idx = None
        self.traj_len = num_steps_loaded

    def pipeline(self, num_traj: int, im_size: int, num_steps_pred: int) -> None:
        save_dir = self._define_save_dir()
        save_dir.mkdir(parents=True, exist_ok=True)

        traj_len = self.traj_len
        H = W = im_size

        if self.drone_dim == 3:
            views = ["left", "right"]
        elif self.drone_dim == 2:
            views = [None]
        else:
            raise ValueError(f"Unsupported drone_dim: {self.drone_dim}")

        V = len(views)

        print(f"[{self.phase}] Conversion of the PNG files in a unique numpy array")
        print(f"[{self.phase}] Raw PNG dir: {self._define_png_dir()}")
        print(f"[{self.phase}] Save dir: {save_dir}")

        im_path = save_dir / "im_dataset_memmap.dat"

        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="w+",
            shape=(num_traj, traj_len, V, H, W),
        )

        name_png_dir = self._define_png_dir()

        def worker_load(args):
            i, j, v_idx, path = args
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            # if img is None:
            #     raise RuntimeError(f"Image not found: {path}")

            img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            return i, j, v_idx, img_resized

        paths = []
        for i in range(num_traj):
            for j in range(traj_len):
                for v_idx, v in enumerate(views):
                    if v is None:
                        p = name_png_dir / f"traj_{i}" / f"step_{j}.png"
                    else:
                        p = name_png_dir / f"traj_{i}" / f"{v}" / f"step_{j}.png"

                    paths.append((i, j, v_idx, p))

        nb_threads = min(8, os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=nb_threads) as executor:
            futures = {executor.submit(worker_load, tup): tup[:3] for tup in paths}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Chargement"):
                i, j, v_idx = futures[future]

                try:
                    _, _, _, img_proc = future.result()
                    im_dataset[i, j, v_idx, :, :] = img_proc

                except Exception as exc:
                    print(
                        f"Erreur traitement image "
                        f"traj={i}, step={j}, view={v_idx}: {exc}"
                    )

        im_dataset.flush()
        del im_dataset

        self.reorganise_dataset_mmap(
            im_path=im_path,
            num_simulations=num_traj,
            traj_len=traj_len,
            seq_len=num_steps_pred,
            resolution=im_size,
            save_dir=save_dir,
            num_views=V,
            delay=self.params.postprocessing.delay,
        )

    def _define_save_dir(self) -> Path:
        return self.dataset_paths.vision_split_dir(self.phase)

    def _define_png_dir(self) -> Path:
        """
        Raw images associated with the same data_generation root.

        Recommended layout:
            datasets/{drone_dim}d/{dataset_version}/raw_images/{phase}/...
        """
        return self.dataset_paths.root / "raw_images" / self.phase

    def _get_num_steps_loaded_for_phase(self, phase: str) -> int:
        if phase == "train":
            return self.params.train.num_steps_loaded

        for idx, spec in enumerate(self.params.val_datasets, start=1):
            default_name = f"val_{idx}"
            name = spec.name or default_name

            if name == phase or spec.split == phase:
                return spec.num_steps_loaded

        raise ValueError(f"Unknown phase: {phase}")

    def reorganise_dataset_mmap(
        self,
        im_path: Path,
        num_simulations: int,
        traj_len: int,
        seq_len: int,
        resolution: int,
        save_dir: Path,
        num_views: int = 1,
        delay: int = 1,
    ) -> None:
        if delay < 1:
            raise ValueError(f"delay must be >= 1, got {delay}")

        if traj_len <= delay:
            raise ValueError(
                f"traj_len={traj_len} must be > delay={delay} to build windows"
            )

        print("Dataset reshape (memmap / par trajectoire)")

        total_samples, num_seq, H, W, usable_samples_per_traj = self._compute_shapes(
            num_simulations=num_simulations,
            traj_len=traj_len,
            seq_len=seq_len,
            resolution=resolution,
            delay=delay,
        )

        im_dataset = np.memmap(
            im_path,
            dtype=np.uint8,
            mode="r",
            shape=(num_simulations, traj_len, num_views, H, W),
        )

        C = (delay + 1) * num_views

        y_sliced = np.memmap(
            save_dir / "dataset_memmap.dat",
            dtype=np.uint8,
            mode="w+",
            shape=(num_seq, seq_len, C, H, W),
        )

        y_flat = y_sliced.reshape(total_samples, C, H, W)

        pair_idx = 0

        for i in tqdm(range(num_simulations), desc="Reshape traj-by-traj"):
            traj = im_dataset[i]
            windows = self._traj_to_windows(traj, delay)

            # Keep only a multiple of seq_len inside each trajectory.
            # This avoids mixing the end of one trajectory with the beginning of the next.
            windows = windows[:usable_samples_per_traj]

            n_i = windows.shape[0]
            y_flat[pair_idx:pair_idx + n_i] = windows
            pair_idx += n_i

        y_sliced.flush()

        self._write_metadata(
            save_dir=save_dir,
            y_shape=[num_seq, seq_len, C, H, W],
            n_views=num_views,
            delay=delay,
            source_png_dir=self._define_png_dir(),
            dataset_root=self.dataset_paths.root,
        )

        del y_sliced, y_flat, im_dataset
        os.remove(im_path)

    @staticmethod
    def _compute_shapes(
            num_simulations: int,
            traj_len: int,
            seq_len: int,
            resolution: int,
            delay: int,
    ) -> tuple[int, int, int, int, int]:
        n_samples_per_traj = traj_len - delay

        if n_samples_per_traj <= 0:
            raise ValueError(
                f"Invalid temporal config: traj_len={traj_len}, delay={delay}. "
                "Expected traj_len > delay."
            )

        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")

        num_seq_per_traj = n_samples_per_traj // seq_len

        if num_seq_per_traj == 0:
            raise ValueError(
                f"Cannot build one sequence: traj_len={traj_len}, "
                f"delay={delay}, n_samples_per_traj={n_samples_per_traj}, "
                f"seq_len={seq_len}."
            )

        usable_samples_per_traj = num_seq_per_traj * seq_len
        total_samples = num_simulations * usable_samples_per_traj
        num_seq = num_simulations * num_seq_per_traj
        H = W = resolution

        return total_samples, num_seq, H, W, usable_samples_per_traj

    @staticmethod
    def _traj_to_windows(traj: np.ndarray, delay: int) -> np.ndarray:
        """
        traj: (T, V, H, W)
        returns: (T-delay, (delay+1)*V, H, W)

        channel layout:
            concat([views@t-delay, ..., views@t])
        """
        T = traj.shape[0]
        chunks = [traj[o: T - delay + o] for o in range(delay + 1)]
        return np.concatenate(chunks, axis=1)

    @staticmethod
    def _write_metadata(
        save_dir: Path,
        y_shape,
        n_views: int,
        delay: int,
        source_png_dir: Path,
        dataset_root: Path,
    ) -> None:
        parts = [f"views@t-{k}" for k in range(delay, 0, -1)] + ["views@t"]
        channel_layout = "concat([" + ", ".join(parts) + "])"

        metadata = {
            "y_shape": list(y_shape),
            "dtype": "uint8",
            "n_views": int(n_views),
            "delay": int(delay),
            "channel_layout": channel_layout,
            "source_png_dir": str(source_png_dir),
            "dataset_root": str(dataset_root),
        }

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)