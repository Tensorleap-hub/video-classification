import os
import yaml
from typing import Any, Dict

def load_yaml(path) -> Dict[str, Any]:
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, path)
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_project_config() -> Dict[str, Any]:

    config = load_yaml("leap_config.yaml")
    config["clip_duration"] = (
                                config["model_transform_params"]["num_frames"] * config["model_transform_params"]["sampling_rate"]
                              ) / config["frames_per_second"]
    return config

CONFIG = load_project_config()