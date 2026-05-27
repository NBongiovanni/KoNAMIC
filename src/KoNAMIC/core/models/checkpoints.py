import joblib
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from KoNAMIC.core import utils
from .vision_koop_model import VisionKoopModel
from .factory import (
    VisionModelAndTC,
    SensorModelAndScaler,
    _make_optimizer_vision,
    _build_vision_model,
    _build_sensor_model
)


def load_vision_koop_model_for_train(
        model_params: dict,
        training_params: dict,
        epoch: int,
        run_dir: Path
) -> VisionModelAndTC:
    model, device = _build_vision_model(model_params)

    ckpt_path = utils.build_checkpoint_path(run_dir, epoch)
    ckpt = _load_ckpt_state_dict_vision(model, str(ckpt_path), device)

    optimizer = _make_optimizer_vision(training_params["optimizer"], model)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    scheduler = StepLR(optimizer, step_size=1, gamma=training_params["lr_decay"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    writer = SummaryWriter(log_dir=str(training_params["log_dir"]))
    print(optimizer.param_groups[0]["lr"])
    model.train()
    return model, (optimizer, scheduler, writer), ckpt


def load_koop_model_for_eval(modality: str, model_params, epoch: int, run_dir: Path):
    if modality == "vision":
        koop_model, x_scaler, u_scaler = load_vision_koop_model_for_eval(
            model_params,
            epoch,
            run_dir,
        )
    elif modality == "sensor":
        koop_model, x_scaler, u_scaler = load_sensor_koop_model_for_eval(
            model_params,
            epoch,
            run_dir,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return koop_model, x_scaler, u_scaler


def load_vision_koop_model_for_eval(
        model_params: dict,
        epoch: int,
        run_dir: Path
):
    model, device = _build_vision_model(model_params)
    ckpt_path = utils.build_checkpoint_path(run_dir, epoch)
    _load_ckpt_state_dict_vision(model, str(ckpt_path), device=device)
    model.eval()
    u_scaler = joblib.load(run_dir / "u_scaler.pkl")
    x_scaler = None
    return model, x_scaler, u_scaler


def load_sensor_koop_model_for_eval(
        model_params: dict,
        epoch: int,
        run_dir: Path,
        pruning_started: bool = False,
) -> SensorModelAndScaler:
    model, device = _build_sensor_model(model_params)

    ckpt_path = utils.build_checkpoint_path(run_dir, epoch)
    checkpoint = torch.load(ckpt_path, map_location=device)
    sd = checkpoint.get("model_state_dict", {})

    print("ckpt_path:", ckpt_path)
    print("type(checkpoint):", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("top-level keys:", list(checkpoint.keys())[:20])

    sd = checkpoint.get("model_state_dict", {})
    print("type(sd):", type(sd))
    if isinstance(sd, dict):
        print("sd keys:", list(sd.keys())[:20])

    model.load_state_dict(sd, strict=True)

    model.eval()
    u_scaler = joblib.load(run_dir / "u_scaler.pkl")
    x_scaler = joblib.load(run_dir / "x_scaler.pkl")
    return model, x_scaler, u_scaler

def _remap_old_dyn_keys(sd: dict) -> dict:
    # ancien -> nouveau
    if "z_drift.weight" in sd and "dyn.z_drift.weight" not in sd:
        sd = sd.copy()
        sd["dyn.z_drift.weight"] = sd.pop("z_drift.weight")
    if "z_act.weight" in sd and "dyn.z_act.weight" not in sd:
        sd = sd.copy()
        sd["dyn.z_act.weight"] = sd.pop("z_act.weight")
    return sd


def _load_ckpt_state_dict_vision(model: VisionKoopModel, ckpt_path: str, device: torch.device):
    """
    Load model weights from a checkpoint path.

    Args:
        model (VisionKoopModel): Target model.
        ckpt_path (str): Full path to the checkpoint file.
        device (torch.device): Device to map tensors to.
        strict (bool): Whether to enforce an exact key match.

    Returns:
        dict: The full checkpoint dictionary loaded from disk.

    Notes:
        - Prints missing/unexpected keys when `strict=False` to aid debugging.
    """

    def _install_legacy_aliases() -> None:
        import sys
        import KoNAMIC

        # Alias package racine
        sys.modules.setdefault("KSIC_v6", KoNAMIC)

        # Optionnel: si tu vois ensuite une erreur sur un sous-module,
        # il faudra aussi aliaser précisément, ex:
        # import KSIC_v8.model_learning as ml
        # sys.modules["KSIC_v6.model_learning"] = ml

    _install_legacy_aliases()
    ckpt = torch.load(ckpt_path, map_location=device)

    sd = ckpt.get("model_state_dict", {})

    # 1) Remap "old checkpoints" keys -> new keys (dyn.*)
    sd = _remap_old_dyn_keys(sd)

    # 2) (optionnel) si pruning: dé-prune AVANT de charger
    #    (utile si tu as déjà eu des ckpt avec weight_orig/weight_mask)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        if _is_pruning_key_mismatch(e):
            sd_clean = _de_prune_state_dict(sd)
            missing, unexpected = model.load_state_dict(sd_clean, strict=True)
        else:
            raise

    if missing or unexpected:
        print(f"[load] missing_keys={missing} | unexpected_keys={unexpected}")
    return ckpt

def _is_pruning_key_mismatch(err: RuntimeError) -> bool:
    msg = str(err)
    return (
        "weight_orig" in msg and "weight_mask" in msg
        and ("Missing key(s)" in msg or "Unexpected key(s)" in msg)
    )


def _de_prune_state_dict(state_dict: dict) -> dict:
    """
    Convert a pruned PyTorch state_dict (weight_orig + weight_mask) into a standard one (weight).
    Works for any module where pruning was applied on 'weight'.
    """
    sd = dict(state_dict)  # shallow copy is fine (tensors not copied)
    keys = list(sd.keys())
    for k in keys:
        if k.endswith("weight_orig"):
            base = k[:-len("_orig")]      # e.g. "z_drift.weight"
            mask_k = base + "_mask"       # e.g. "z_drift.weight_mask"
            if mask_k in sd:
                sd[base] = sd[k] * sd[mask_k]
                del sd[k]
                del sd[mask_k]
    return sd
