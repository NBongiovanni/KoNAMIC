import os
import numpy as np
import random
import  torch


def set_seed_light(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed(seed: int, deterministic: bool = False) -> None:
    # Python / NumPy
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Pour reproductibilité complète (hash Python)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Désactive certaines optimisations non déterministes
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # (optionnel mais recommandé sur versions récentes)
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)