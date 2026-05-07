"""config_loader.py — YAML 配置加载 + type 三段式解析"""
import os
import glob
import yaml
from datetime import datetime, timedelta
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
    "output"         → TypeInfo(base="出力", sub=None, value_type="实际")  # 英文向后兼容

    默认假设: 不显式声明 value_type 就当作 实际.
    """
    # 英文向后兼容映射
    en_to_cn = {"output": "出力", "load": "负荷", "price": "电价"}
    s = en_to_cn.get(type_str, type_str)

    cfg = load_config()
    # 检查 type_str 是否就是基类本身或映射后的基类
    if s in get_base_types():
        return TypeInfo(base=s, sub=None, value_type="实际")

    parts = s.rsplit("_", 2)
    value_type_values = set(cfg.get("value_types", {}).values())

    # 末段是 value_type?
    if len(parts) >= 2 and parts[-1] in value_type_values:
        base = parts[0]
        sub = "_".join(parts[1:-1]) if len(parts) > 2 else None
        return TypeInfo(base=base, sub=sub or None, value_type=parts[-1])

    # 只有 base_sub 格式，value_type 默认 实际
    if len(parts) >= 2:
        return TypeInfo(base=parts[0], sub="_".join(parts[1:]), value_type="实际")

    return TypeInfo(base=parts[0], sub=None, value_type="实际")


def type_info_to_str(ti: TypeInfo, include_value_type: bool = True) -> str:
    """TypeInfo → type 字符串."""
    parts = [ti.base]
    if ti.sub:
        parts.append(ti.sub)
    if include_value_type:
        parts.append(ti.value_type)
    return "_".join(parts)


def type_info_to_key(ti: TypeInfo) -> str:
    """TypeInfo → model key 用的字符串 (base_sub, 不含 value_type)."""
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
    return _CONFIG.get("mysql", {})


def get_doris_config() -> Dict[str, Any]:
    return _CONFIG.get("doris", {})


def get_model_config() -> Dict[str, Any]:
    return _CONFIG.get("model", {})


def get_provinces() -> list:
    return _CONFIG.get("provinces", [])


def get_base_types() -> list:
    """返回基类类型列表 (prefix 校验用)."""
    return _CONFIG.get("types", ["出力", "负荷", "电价"])


def get_types() -> list:
    """向后兼容: 返回英文基类名."""
    cn_to_en = {"出力": "output", "负荷": "load", "电价": "price"}
    return [cn_to_en.get(t, t) for t in get_base_types()]


def get_province_coords() -> Dict[str, Dict[str, float]]:
    return _CONFIG.get("province_coords", {})


def get_validator_config() -> Dict[str, Any]:
    return _CONFIG.get("validator", {})


def get_improver_config() -> Dict[str, Any]:
    return _CONFIG.get("improver", {})


def get_type_features(base_type: str, sub_type: Optional[str] = None) -> Dict[str, List[str]]:
    """返回 (base_type, sub_type) 对应的特征需求矩阵."""
    matrix = _CONFIG.get("type_features", {})
    if sub_type:
        key = f"{base_type}_{sub_type}"
    else:
        key = base_type
    return matrix.get(key, {"critical": [], "optional": []})


def get_cross_type_rules() -> Dict[str, Any]:
    return _CONFIG.get("cross_type_rules", {})


def get_available_types(province: str) -> List[str]:
    """从文件系统扫描该省份实际存在的 type 全称列表 (含 value_type).

    文件名格式:
      旧: {province}_{type}_{YYYYMMDD}.parquet  (e.g. 广东_load_20260501.parquet)
      新: {province}_{type}_{YYYYMMDD}.parquet  (e.g. 广东_出力_风电_实际_20260501.parquet)
    日期后缀固定为 8 位数字 (YYYYMMDD).
    """
    # 英文→中文映射 (旧文件向后兼容)
    en_to_cn = {"output": "出力", "load": "负荷", "price": "电价"}

    base_dir = os.path.join(_SKILL_HOME, ".energy_data", "features")
    types_set = set()
    pattern = os.path.join(base_dir, f"{province}_*.parquet")
    for f in sorted(glob.glob(pattern)):
        fname = os.path.basename(f)
        # 去掉 "{province}_" 前缀和 ".parquet" 后缀
        rest = fname[len(province) + 1:].replace(".parquet", "")
        # rest 格式: type_YYYYMMDD 或 type_sub_vt_YYYYMMDD

        # 去掉末尾 8 位日期
        parts = rest.split("_")
        if parts and len(parts[-1]) == 8 and parts[-1].isdigit():
            parts = parts[:-1]  # 去掉日期
        type_str = "_".join(parts) if parts else ""

        # 旧英文名映射为中文
        if type_str in en_to_cn:
            type_str = en_to_cn[type_str]

        if type_str:
            types_set.add(type_str)

    if not types_set:
        # 回退: 扫描 raw 目录
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
    """只返回 value_type=实际 的 type."""
    all_types = get_available_types(province)
    cfg = load_config()
    vt_actual = set(cfg.get("value_types", {}).values())
    return [t for t in all_types if not any(t.endswith(f"_{v}") for v in vt_actual) or t.endswith("_实际")]


def get_data_delay(province: str, target_type: str) -> int:
    """返回该 province/type 的数据可用延迟天数.

    负值 = 提前可用 (如日前预测 D+1 → -1)
    正值 = 延迟到库 (如实际值 D-1 → 1, 结算 D-6 → 6)
    0 = 当天可用.

    匹配优先级: province_overrides > type_overrides > default_delays(value_type)
    """
    cfg = load_config()
    da_cfg = cfg.get("data_availability", {})
    if not da_cfg:
        return 0

    ti = parse_type(target_type)
    base_sub = f"{ti.base}_{ti.sub}" if ti.sub else ti.base

    # 1) 省份级覆盖 (最高优先级)
    prov_overrides = da_cfg.get("province_overrides", {})
    if province in prov_overrides and base_sub in prov_overrides[province]:
        return int(prov_overrides[province][base_sub])

    # 2) 类型级覆盖
    type_overrides = da_cfg.get("type_overrides", {})
    if base_sub in type_overrides:
        return int(type_overrides[base_sub])

    # 3) 默认规则 (按 value_type 后缀)
    default_delays = da_cfg.get("default_delays", {})
    for vt_suffix, delay in default_delays.items():
        if ti.value_type == vt_suffix:
            return int(delay)

    return 0


def get_available_date(province: str, target_type: str,
                       reference_date: Optional[datetime] = None) -> datetime:
    """返回该 province/type 在 reference_date 时实际可用的最新数据日期.

    reference_date: 运行日期 (默认今天)
    返回: reference_date - delay_days
    """
    from datetime import date as _date_type
    delay = get_data_delay(province, target_type)
    ref = reference_date if reference_date is not None else datetime.now()
    if isinstance(ref, _date_type) and not isinstance(ref, datetime):
        ref = datetime.combine(ref, datetime.min.time())
    return ref - timedelta(days=delay)


def validate_province_and_type(province: str, target_type: str) -> None:
    """校验 province 和 type. type 支持三段式 (前缀匹配基类)."""
    valid_provinces = get_provinces()
    if province not in valid_provinces:
        raise ValueError(f"未知省份 '{province}'，合法值: {valid_provinces}")

    ti = parse_type(target_type)
    base_types = get_base_types()
    if ti.base not in base_types:
        raise ValueError(f"未知基类类型 '{ti.base}'，合法值: {base_types}")
    if ti.sub is not None and ti.sub.strip() == "":
        raise ValueError(f"子类型不能为空: '{target_type}'")


load_config()
