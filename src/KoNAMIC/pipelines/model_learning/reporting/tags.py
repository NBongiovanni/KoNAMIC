VISION_TAGS = {
    "reconstruction": {
        "y_rec": "reconstruction/01_y_rec",
    },
    "open_loop": {
        "y_pred": "open_loop/01_y_pred",
        "z": "open_loop/02_z",
        "c": "open_loop/03_centroid",
        "angle": "open_loop/04_angle",
        "horizontal": "open_loop/05_horizontal",
        "vertical": "open_loop/06_vertical",
        "iou": "open_loop/07_iou",
        "state_rmse": "open_loop/08_state_rmse",
    },
    "closed_loop": {
        "y": "closed_loop/01_state",
        "z": "closed_loop/02_z",
        "state_rmse": "closed_loop/03_state_rmse",
    },
}

SENSOR_TAGS = {
    "reconstruction": {
        "y_rec": "reconstruction/01_y_rec",
    },
    "open_loop": {
        "y_pred": "open_loop/01_y_pred",
        "z_pred": "open_loop/02_z_pred",
        "state_rmse": "open_loop/03_state_rmse",
    },
    "closed_loop": {
        "y": "closed_loop/01_state",
        "z": "closed_loop/02_z",
        "state_rmse": "closed_loop/03_state_rmse",
    },
}


def build_state_error_tags(prefix: str, state_labels: list[str]) -> dict[str, str]:
    tags = {}

    for idx, label in enumerate(state_labels, start=1):
        clean_label = (
            label.replace("'", "dot")
            .replace(" ", "_")
            .replace("/", "_")
        )
        tags[label] = f"{prefix}/{idx:02d}_{clean_label}_error"

    return tags


def get_error_state_labels(modality: str, system) -> list[str]:
    if modality == "vision":
        return system.get_x_labels(only_positions=True)

    if modality == "sensor":
        return system.get_x_labels(only_positions=False)

    raise ValueError(f"Unknown modality: {modality}")


def build_closed_loop_summary_tags(tags: dict[str, dict[str, str]]) -> dict[str, str]:
    y_tag = tags["closed_loop"]["y"]
    z_tag = tags["closed_loop"]["z"]
    return {
        "x_rmse": f"{y_tag}",
        "z_rmse": f"{z_tag}",
    }


def get_tags(modality: str) -> dict[str, dict[str, str]]:
    if modality == "vision":
        return VISION_TAGS
    if modality == "sensor":
        return SENSOR_TAGS
    raise ValueError(f"Unknown modality: {modality}")
