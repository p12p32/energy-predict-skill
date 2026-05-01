"""config_loader.py — YAML 配置加载与访问"""
import os
import yaml
from typing import Any, Dict

_CONFIG: Dict[str, Any] = {}
_SKILL_HOME = os.environ.get("ENERGY_HOME", os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def load_config(config_path: str = None) -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    if config_path is None:
        config_path = os.path.join(_SKILL_HOME, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    if "doris" in _CONFIG:
        _CONFIG["doris"]["host"] = os.environ.get("DORIS_HOST", _CONFIG["doris"]["host"])
        _CONFIG["doris"]["port"] = int(os.environ.get("DORIS_PORT", _CONFIG["doris"]["port"]))
        _CONFIG["doris"]["user"] = os.environ.get("DORIS_USER", _CONFIG["doris"]["user"])
        _CONFIG["doris"]["password"] = os.environ.get("DORIS_PASSWORD", _CONFIG["doris"]["password"])
        _CONFIG["doris"]["database"] = os.environ.get("DORIS_DATABASE", _CONFIG["doris"]["database"])

    return _CONFIG


def get_doris_config() -> Dict[str, Any]:
    return _CONFIG.get("doris", {})


def get_model_config() -> Dict[str, Any]:
    return _CONFIG.get("model", {})


def get_provinces() -> list:
    return _CONFIG.get("provinces", [])


def get_types() -> list:
    return _CONFIG.get("types", ["output", "load"])


def get_province_coords() -> Dict[str, Dict[str, float]]:
    return _CONFIG.get("province_coords", {})


def get_validator_config() -> Dict[str, Any]:
    return _CONFIG.get("validator", {})


def get_improver_config() -> Dict[str, Any]:
    return _CONFIG.get("improver", {})


load_config()
