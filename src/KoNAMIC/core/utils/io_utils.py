from typing import Mapping

from scipy.io import savemat, loadmat
import torch


def losses_to_jsonable(losses: Mapping[str, torch.Tensor]) -> dict:
    out = {}
    for k, v in losses.items():
        if v is None:
            out[k] = None
        elif isinstance(v, torch.Tensor):
            if v.numel() == 0:
                out[k] = None  # ou 0.0 si tu préfères
            else:
                # moyenne au cas où la loss serait vectorielle
                out[k] = float(v.detach().mean().item())
        else:
            out[k] = float(v)
    return out


def save_array_for_matlab(filename: str, array_dict: dict):
    """
    Sauvegarde un ou plusieurs numpy arrays dans un fichier .mat lisible par Matlab.

    Exemple d'appel :
        save_array_for_matlab("data.mat", {"X": X, "Y": Y})
    """
    savemat(filename, array_dict)
    print(f"✅ Données enregistrées dans {filename}")


def load_array_from_matlab(filename: str) -> dict:
    """
    Charge un fichier .mat (retourne un dictionnaire avec les variables Matlab).

    Exemple d'appel :
        data = load_array_from_matlab("data.mat")
        X = data["X"]
    """
    data = loadmat(filename)
    print(f"📂 Données chargées depuis {filename}")
    return data