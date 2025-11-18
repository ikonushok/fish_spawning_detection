# src/config_loader.py
from pathlib import Path
import yaml


def load_config(config_path: str | Path = "../configs/config.yaml") -> dict:
    config_path = Path(config_path)
    with config_path.open("r") as f:
        return yaml.safe_load(f)
