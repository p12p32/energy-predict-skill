"""daemon.py — 单次调度入口 (cron / AI 按需调用)

修复:
- 移除 MonitoringServer (阻塞根源)
- 纯单次执行: build → train → predict → validate → 退出
- 全局超时保护 30 分钟
- 每步计时 + 进度日志
- 异常不中断
"""
import logging
import os
import sys
import time
from datetime import datetime

if "--crontab" in sys.argv:
    script = os.path.abspath(__file__)
    proj_dir = os.path.dirname(os.path.dirname(__file__))
    print("# 能源预测调度 — crontab 模板")
    print(f"# 日常增量运行 (每天凌晨 6:00)")
    print(f"0 6 * * * cd {proj_dir} && python {script}")
    print()
    print(f"# 周度全量重建 (每周日 4:00)")
    print(f"0 4 * * 0 cd {proj_dir} && python {script} --force-rebuild")
    sys.exit(0)

from scripts.orchestrator import Orchestrator
from scripts.core.config import get_provinces, get_available_types, get_daemon_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("energy_daemon.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daemon")


class Daemon:
    def __init__(self):
        self.orch = Orchestrator()
        self.force_rebuild_features: bool = False
        self.last_mape: dict = {}
        self._start_time = 0

    def run(self):
        cfg = get_daemon_config()
        timeout = cfg.get("timeout_seconds", 1800)
        self._start_time = time.time()

        logger.info("=" * 50)
        logger.info(f"能源预测引擎 启动 (超时: {timeout}s)")
        logger.info("=" * 50)

        try:
            # 1. 构建特征
            self._step("特征构建", lambda: self.orch.build_features(
                force_rebuild=self.force_rebuild_features
            ))

            self._check_timeout(timeout)

            # 2. 扫描 + 训练
            available = self.orch._scan_available()
            if not available:
                logger.warning("无可用数据, 跳过训练")
                return

            self._step("模型训练", lambda: self.orch.train_all())
            self._check_timeout(timeout)

            # 3. 预测 + 验证
            self._predict_and_validate()

            # 4. 自动优化
            self._step("自动优化", self._auto_improve_cycle)

        except TimeoutError:
            logger.error("全局超时 (%ds), 提前退出", timeout)
        except Exception as e:
            logger.error("运行异常: %s", e, exc_info=True)

        self._print_summary()

    def _step(self, name: str, fn):
        t0 = time.time()
        logger.info(f"[开始] {name}")
        try:
            result = fn()
            elapsed = time.time() - t0
            logger.info(f"[完成] {name} ({elapsed:.1f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"[失败] {name} ({elapsed:.1f}s): {e}")
            return None

    def _check_timeout(self, timeout: int):
        elapsed = time.time() - self._start_time
        if elapsed > timeout:
            raise TimeoutError(f"已超时 ({elapsed:.0f}s > {timeout}s)")

    def _predict_and_validate(self):
        for province in get_provinces():
            for target_type in get_available_types(province):
                key = f"{province}_{target_type}"
                try:
                    result = self.orch.predict(province, target_type, 24)
                    health = result.get("health", {})
                    mape = health.get("mape", 0)
                    status = health.get("status", "ok")
                    logger.info(f"[{key}] 预测完成: MAPE={mape}, status={status}")

                    if mape:
                        self.last_mape[key] = mape

                    # 验证 + 自动优化
                    val = self.orch.run_validation_cycle(province, target_type)
                    if val.get("status") == "improved":
                        imp = val.get("improvement", {})
                        logger.info(f"[{key}] 自动优化成功: "
                                    f"MAPE {imp.get('mape_before')}→{imp.get('mape_after')}")
                    elif "report" in val and val["report"].get("metrics", {}).get("mape"):
                        self.last_mape[key] = val["report"]["metrics"]["mape"]

                except Exception as e:
                    logger.error(f"[{key}] 异常: {e}")

    def _auto_improve_cycle(self):
        """对 MAPE 最差的几个类型触发自动优化."""
        if not self.last_mape:
            return

        # 取 MAPE 最差的 3 个
        worst = sorted(self.last_mape.items(), key=lambda x: x[1] or 999, reverse=True)[:3]
        for key, mape in worst:
            if mape is None or mape < 0.05:
                continue
            parts = key.split("_", 1)
            if len(parts) != 2:
                continue
            province, target_type = parts[0], parts[1]
            logger.info(f"触发自动优化: {key} (MAPE={mape})")
            try:
                self.orch.auto_improve(province, target_type)
            except Exception as e:
                logger.warning(f"自动优化失败 {key}: {e}")

    def _print_summary(self):
        elapsed = time.time() - self._start_time
        logger.info("=" * 50)
        logger.info(f"本轮完成 ({elapsed:.1f}s)")
        for key, mape in sorted(self.last_mape.items(), key=lambda x: x[1] or 999):
            logger.info(f"  {key}: MAPE={mape}")
        logger.info("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="能源预测调度引擎")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--crontab", action="store_true")
    args = parser.parse_args()

    d = Daemon()
    d.force_rebuild_features = args.force_rebuild
    d.run()
