import logging
import sys


def setup_logging() -> logging.Logger:
    # Configuration globale du logging
    logging.basicConfig(
        level=logging.INFO,                                # niveau minimal de messages
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),             # envoi vers la sortie standard
        ]
    )
    logger = logging.getLogger(__name__)
    return logger