"""Resolve a policy configuration in its isolated workspace, not the simulator's."""
from pathlib import Path
import yaml


def read_policy_config(path, workspace):
    def resolve(value):
        if isinstance(value, str):
            return value.replace('${ROOT}', str(workspace))
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value
    return resolve(yaml.safe_load(Path(path).read_text()))
