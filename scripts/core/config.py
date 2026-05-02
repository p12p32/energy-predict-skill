"""config_loader.py — YAML 配置加载 + type 三段式解析"""
import os
import glob
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_CONFIG: Dict[str, Any] = {}
_SKILL_HOME = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


@dataclass
class TypeInfo:
    """三段式 type 解析结果."""
    base: str           # 出力 | 负荷 | 电价
    sub: Optional[str]  # 风电 | 光伏 | 水电 | 工业 | 日前 | ... (None=无子类型)
    value_type: str     # 实际 | 预测 (默认 实际)


def parse_type(type_str: str) -> TypeInfo:
    """解析三段式 type 字符串.

    "出力_风电_实际" → TypeInfo(base="出力", sub="风电", value_type="实际")
    "出力_风电"      → TypeInfo(base="出力", sub="风电", value_type="实际")
    "出力"           → TypeInfo(base="出力", sub=None, value_type="实际")
    "电价_日前_预测"  → TypeInfo(base="电价", sub="日前", value_type="预测")
    "output"         → TypeInfo(base="出力", sub=None, value_type="实际")
    """
    en_to_cn = {"output": "出力", "load": "负荷", "price": "电价"}
    s = en_to_cn.get(type_str, type_str)

    cfg = load_config()
    if s in get_base_types():
        return TypeInfo(base=s, sub=None, value_type="实际")

    parts = s.rsplit("_", 2)
    value_type_values = set(cfg.get("value_types", {}).values())

    if len(parts) >= 2 and parts[-1] in value_type_values:
        base = parts[0]
        sub = "_".join(parts[1:-1]) if len(parts) > 2 else None
        return TypeInfo(base=base, sub=sub or None, value_type=parts[-1])

    if len(parts) >= 2:
        return TypeInfo(base=parts[0], sub="_".join(parts[1:]), value_type="实际")

    return TypeInfo(base=parts[0], sub=None, value_type="实际")


def type_info_to_str(ti: TypeInfo, include_value_type: bool = True) -> str:
    parts = [ti.base]
    if ti.sub:
        parts.append(ti.sub)
    if include_value_type:
        parts.append(ti.value_type)
    return "_".join(parts)


def type_info_to_key(ti: TypeInfo) -> str:
    parts = [ti.base]
    if ti.sub:
        parts.append(ti.sub)
    return "_".join(parts)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    global _CONFIG
    if _CONFIG:
        return _CONFIG

    if config_path is None:
        config_path = os.path.join(_SKILL_HOME, "assets", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = yaml.safe_load(f)

    if "doris" in _CONFIG:
        _CONFIG["doris"]["host"] = os.environ.get("DORIS_HOST", _CONFIG["doris"]["host"])
        _CONFIG["doris"]["port"] = int(os.environ.get("DORIS_PORT", _CONFIG["doris"]["port"]))
        _CONFIG["doris"]["user"] = os.environ.get("DORIS_USER", _CONFIG["doris"]["user"])
        _CONFIG["doris"]["password"] = os.environ.get("DORIS_PASSWORD", _CONFIG["doris"]["password"])
        _CONFIG["doris"]["database"] = os.environ.get("DORIS_DATABASE", _CONFIG["doris"]["database"])

    return _CONFIG


def get_mysql_config() -> Dict[str, Any]:
    return load_config().get("mysql", {})


def get_doris_config() -> Dict[str, Any]:
    return load_config().get("doris", {})


def get_model_config() -> Dict[str, Any]:
    return load_config().get("model", {})


def get_training_config() -> Dict[str, Any]:
    return load_config().get("training", {})


def get_cache_config() -> Dict[str, Any]:
    return load_config().get("cache", {})


def get_prediction_config() -> Dict[str, Any]:
    return load_config().get("prediction", {})


def get_daemon_config() -> Dict[str, Any]:
    return load_config().get("daemon", {})


def get_provinces() -> list:
    return load_config().get("provinces", [])


def get_base_types() -> list:
    return load_config().get("types", ["出力", "负荷", "电价"])


def get_types() -> list:
    cn_to_en = {"出力": "output", "负荷": "load", "电价": "price"}
    return [cn_to_en.get(t, t) for t in get_base_types()]


def get_province_coords() -> Dict[str, Dict[str, float]]:
    return load_config().get("province_coords", {})


def get_validator_config() -> Dict[str, Any]:
    return load_config().get("validator", {})


def get_improver_config() -> Dict[str, Any]:
    return load_config().get("improver", {})


def get_type_features(base_type: str, sub_type: Optional[str] = None) -> Dict[str, List[str]]:
    matrix = load_config().get("type_features", {})
    if sub_type:
        key = f"{base_type}_{sub_type}"
    else:
        key = base_type
    return matrix.get(key, {"critical": [], "optional": []})


def get_cross_type_rules() -> Dict[str, Any]:
    return load_config().get("cross_type_rules", {})


def get_available_types(province: str) -> List[str]:
    en_to_cn = {"output": "出力", "load": "负荷", "price": "电价"}
    base_dir = os.path.join(_SKILL_HOME, ".energy_data", "features")
    types_set = set()
    pattern = os.path.join(base_dir, f"{province}_*.parquet")
    for f in sorted(glob.glob(pattern)):
        fname = os.path.basename(f)
        rest = fname[len(province) + 1:].replace(".parquet", "")
        parts = rest.split("_")
        if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
            parts = parts[:-1]
        type_str = "_".join(parts) if parts else ""
        if type_str in en_to_cn:
            type_str = en_to_cn[type_str]
        if type_str:
            types_set.add(type_str)

    if not types_set:
        raw_dir = os.path.join(_SKILL_HOME, ".energy_data", "raw")
        raw_pattern = os.path.join(raw_dir, f"{province}_*.csv")
        for f in sorted(glob.glob(raw_pattern)):
            fname = os.path.basename(f).replace(".csv", "")
            rest = fname[len(province) + 1:] if fname.startswith(f"{province}_") else fname
            if rest in en_to_cn:
                rest = en_to_cn[rest]
            if rest:
                types_set.add(rest)

    return sorted(types_set)


def get_available_actual_types(province: str) -> List[str]:
    all_types = get_available_types(province)
    cfg = load_config()
    vt_actual = set(cfg.get("value_types", {}).values())
    return [t for t in all_types if not any(t.endswith(f"_{v}") for v in vt_actual) or t.endswith("_实际")]


def validate_province_and_type(province: str, target_type: str) -> None:
    valid_provinces = get_provinces()
    if province not in valid_provinces:
        raise ValueError(f"未知省份 '{province}'，合法值: {valid_provinces}")

    ti = parse_type(target_type)
    base_types = get_base_types()
    if ti.base not in base_types:
        raise ValueError(f"未知基类类型 '{ti.base}'，合法值: {base_types}")
    if ti.sub is not None and ti.sub.strip() == "":
        raise ValueError(f"子类型不能为空: '{target_type}'")


# 强制重新加载配置
def reload_config():
    global _CONFIG
    _CONFIG = {}


load_config()
