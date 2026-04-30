"""holiday_utils.py — 中国节假日识别与周期编码"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Set, List


# ============================================================
# 中国法定节假日（2023-2027 硬编码，覆盖常见历史数据范围）
# ============================================================

CHINESE_HOLIDAYS: Dict[int, Dict[str, List[str]]] = {
    2023: {
        "spring_festival": ["2023-01-21", "2023-01-22", "2023-01-23",
                           "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27"],
        "qingming": ["2023-04-05"],
        "labor_day": ["2023-04-29", "2023-04-30", "2023-05-01", "2023-05-02", "2023-05-03"],
        "dragon_boat": ["2023-06-22", "2023-06-23", "2023-06-24"],
        "mid_autumn_national": ["2023-09-29", "2023-09-30",
                                "2023-10-01", "2023-10-02", "2023-10-03",
                                "2023-10-04", "2023-10-05", "2023-10-06"],
    },
    2024: {
        "spring_festival": ["2024-02-10", "2024-02-11", "2024-02-12",
                           "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17"],
        "qingming": ["2024-04-04", "2024-04-05", "2024-04-06"],
        "labor_day": ["2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05"],
        "dragon_boat": ["2024-06-08", "2024-06-09", "2024-06-10"],
        "mid_autumn": ["2024-09-15", "2024-09-16", "2024-09-17"],
        "national_day": ["2024-10-01", "2024-10-02", "2024-10-03",
                        "2024-10-04", "2024-10-05", "2024-10-06", "2024-10-07"],
    },
    2025: {
        "spring_festival": ["2025-01-28", "2025-01-29", "2025-01-30",
                           "2025-01-31", "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04"],
        "qingming": ["2025-04-04", "2025-04-05", "2025-04-06"],
        "labor_day": ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05"],
        "dragon_boat": ["2025-05-31", "2025-06-01", "2025-06-02"],
        "mid_autumn_national": ["2025-10-01", "2025-10-02", "2025-10-03",
                                "2025-10-04", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08"],
    },
    2026: {
        "spring_festival": ["2026-02-17", "2026-02-18", "2026-02-19",
                           "2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23"],
        "qingming": ["2026-04-05"],
        "labor_day": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"],
        "dragon_boat": ["2026-06-19", "2026-06-20", "2026-06-21"],
        "mid_autumn": ["2026-09-25"],
        "national_day": ["2026-10-01", "2026-10-02", "2026-10-03",
                        "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07"],
    },
    2027: {
        "spring_festival": ["2027-02-06", "2027-02-07", "2027-02-08",
                           "2027-02-09", "2027-02-10", "2027-02-11", "2027-02-12"],
        "qingming": ["2027-04-05"],
        "labor_day": ["2027-05-01", "2027-05-02", "2027-05-03", "2027-05-04", "2027-05-05"],
        "dragon_boat": ["2027-06-09"],
        "national_day": ["2027-10-01", "2027-10-02", "2027-10-03",
                        "2027-10-04", "2027-10-05", "2027-10-06", "2027-10-07"],
    },
}

# 构建一个快速查找集合
_holiday_set: Set[str] = set()
for _year_dates in CHINESE_HOLIDAYS.values():
    for _dates in _year_dates.values():
        _holiday_set.update(_dates)

# 特殊调休工作日 (周末但上班)
_WORK_WEEKENDS: Set[str] = set([
    "2023-01-28", "2023-01-29", "2023-04-23", "2023-05-06",
    "2023-06-25", "2023-10-07", "2023-10-08",
    "2024-02-04", "2024-02-18", "2024-04-07", "2024-04-28",
    "2024-05-11", "2024-09-14", "2024-09-29", "2024-10-12",
    "2025-01-26", "2025-02-08", "2025-04-27", "2025-09-28",
    "2025-10-11",
    "2026-02-28", "2026-04-25",
    "2027-02-21", "2027-02-28",
])


def is_holiday(dt) -> bool:
    """判断某个日期是否为节假日."""
    if hasattr(dt, "strftime"):
        date_str = dt.strftime("%Y-%m-%d")
    else:
        date_str = str(dt)[:10]
    return date_str in _holiday_set


def is_work_weekend(dt) -> bool:
    """判断是否为调休工作日(周末但上班)."""
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d") in _WORK_WEEKENDS
    return str(dt)[:10] in _WORK_WEEKENDS


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """给 DataFrame 添加节假日相关特征列.

    添加: is_holiday, is_work_weekend, days_to_holiday, days_from_holiday
    """
    if "dt" not in df.columns:
        return df

    result = df.copy()
    result["is_holiday"] = result["dt"].apply(is_holiday)
    result["is_work_weekend"] = result["dt"].apply(is_work_weekend)

    # ── 距离节假日的天数 ──
    dates = sorted(_holiday_set)
    if dates:
        result["days_to_holiday"] = result["dt"].apply(
            lambda d: min(
                (date.fromisoformat(h) - d.date()).days
                for h in dates
                if date.fromisoformat(h) >= d.date()
            ) if any(date.fromisoformat(h) >= d.date() for h in dates) else 365
        )
        result["days_from_holiday"] = result["dt"].apply(
            lambda d: min(
                (d.date() - date.fromisoformat(h)).days
                for h in dates
                if date.fromisoformat(h) <= d.date()
            ) if any(date.fromisoformat(h) <= d.date() for h in dates) else 365
        )

    return result


# ============================================================
# 周期编码
# ============================================================

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """给时间特征添加 sin/cos 编码，让模型理解周期性."""
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    if "day_of_week" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    if "day_of_month" in df.columns:
        df["dom_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
        df["dom_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)

    return df


# ============================================================
# 深度日历特征
# ============================================================

# 寒暑假区间 (近似，中国中小学标准)
SCHOOL_HOLIDAY_RANGES = [
    ("01-15", "02-28"),  # 寒假
    ("07-01", "08-31"),  # 暑假
]


def is_bridge_day(dt) -> bool:
    """判断是否为桥接日: 工作日夹在节假日和周末之间."""
    if hasattr(dt, "strftime"):
        date_str = dt.strftime("%Y-%m-%d")
        d = dt
    else:
        date_str = str(dt)[:10]
        d = date.fromisoformat(date_str)

    weekday = d.weekday()  # 0=Monday
    # 不是周末 且 前后有假日
    if weekday >= 5:
        return False

    prev_day = d - timedelta(days=1)
    next_day = d + timedelta(days=1)

    prev_holiday = prev_day.strftime("%Y-%m-%d") in _holiday_set or prev_day.weekday() >= 5
    next_holiday = next_day.strftime("%Y-%m-%d") in _holiday_set or next_day.weekday() >= 5

    return prev_holiday and next_holiday and not is_holiday(dt)


def is_school_holiday(dt) -> bool:
    """判断是否为寒暑假."""
    if hasattr(dt, "strftime"):
        month_day = dt.strftime("%m-%d")
    else:
        month_day = str(dt)[5:10]

    for start, end in SCHOOL_HOLIDAY_RANGES:
        if start <= month_day <= end:
            return True
    return False


def working_day_type(dt) -> int:
    """返回工作日类型: 0=周一, 1=正常工作日, 2=周五, 3=节前, 4=节后, 5=周末, 6=假日."""
    if is_holiday(dt):
        return 6
    if hasattr(dt, "weekday"):
        wd = dt.weekday()
    else:
        wd = date.fromisoformat(str(dt)[:10]).weekday()

    if wd >= 5:
        return 5

    # 节前/节后三天
    if hasattr(dt, "strftime"):
        date_str = dt.strftime("%Y-%m-%d")
        d = dt.date() if hasattr(dt, "date") else date.fromisoformat(date_str)
    else:
        date_str = str(dt)[:10]
        d = date.fromisoformat(date_str)

    for offset in range(1, 4):
        if (d + timedelta(days=offset)).strftime("%Y-%m-%d") in _holiday_set:
            return 3  # 节前
        if (d - timedelta(days=offset)).strftime("%Y-%m-%d") in _holiday_set:
            return 4  # 节后

    if is_bridge_day(dt):
        return 3  # 桥接日同节前

    if wd == 0:
        return 0  # 周一
    if wd == 4:
        return 2  # 周五
    return 1  # 正常工作日


def add_deep_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加深度日历特征: bridge_day, school_holiday, working_day_type."""
    result = df.copy()
    if "dt" not in result.columns:
        return result

    result["bridge_day"] = result["dt"].apply(is_bridge_day)
    result["school_holiday"] = result["dt"].apply(is_school_holiday)
    result["working_day_type"] = result["dt"].apply(working_day_type)

    return result
