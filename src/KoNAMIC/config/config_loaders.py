from KoNAMIC.paths import find_project_root
from .config_utils import load_yaml


def load_controller_config(controller: str, system_name: str) -> dict:
    """Load the controller YAML config for the given controller and system."""
    project_root = find_project_root()
    config_path = (
            project_root
            / "configs"
            / "components"
            / "controllers"
            / controller
            / f"{system_name}.yaml"
    )
    return load_yaml(config_path)


def load_typed_controller_config_for_system(
    controller: str,
    system_name: str,
    modality: str,
    variant: str,
):
    from KoNAMIC.core.control.config.loader import load_controller_config

    return load_controller_config(
        controller=controller,
        system_name=system_name,
        modality=modality,
        variant=variant,
    )


def load_closed_loop_eval_config(system_name: str) -> dict:
    """Load the evaluation YAML config for the given system."""
    project_root = find_project_root()
    config_path = (
            project_root
            / "configs"
            / "pipelines"
            / "evaluation"
            / "closed_loop"
            / system_name
            / "sensor.yaml"
    )
    return load_yaml(config_path)


def load_typed_closed_loop_eval_config(system_name: str):
    from KoNAMIC.pipelines.closed_loop_simulation.config import ClosedLoopEvalConfig

    return ClosedLoopEvalConfig.from_dict(load_closed_loop_eval_config(system_name))
