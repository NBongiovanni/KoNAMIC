from KoNAMIC.core import utils


def prepare_control_params(
        sys_params: dict,
        control_params: dict,
        epoch: int,
        runs_dir: utils.RunPaths
) -> dict:
    control_params["run_dir"] =  runs_dir.closed_loop_eval_dir
    control_params["control_runs_dir"] = runs_dir.closed_loop_eval_dir
    model_params = sys_params["model_params"]
    control_params["z_dim"] = model_params["z_dynamics"]["z_dim"]
    control_params["z_dynamics_model"] = model_params["z_dynamics"]["model"]
    control_params["epoch"] = epoch
    return control_params
