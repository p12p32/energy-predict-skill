"""daemon.py — 单次调度入口 (cron / AI 按需调用)

设计原则: 不做常驻轮询。日常由 cron 定时触发，AI 在需要时手动调用。
流程: build_features → train_all → predict_all → validate → 退出.
"""
import logging
import os
import sys

# --crontab 模式不依赖项目模块，提前处理
if "--crontab" in sys.argv:
    script = os.path.abspath(__file__)
    proj_dir = os.path.dirname(os.path.dirname(__file__))
    print("# 能源预测调度 — crontab 模板")
    print(f"# 日常增量运行 (每天凌晨 6:00)")
    print(f"0 6 * * * cd {proj_dir} && python {script}")
    print()
    print(f"# 周度全量重建 (每周日 4:00, 合并碎片文件)")
    print(f"0 4 * * 0 cd {proj_dir} && python {script} --force-rebuild")
    print()
    print(f"# 如果数据在凌晨 3:00 到库, 给 3 小时容错; 按实际情况调整")
    sys.exit(0)

from scripts.orchestrator import Orchestrator
from scripts.core.config import get_provinces, get_available_types, load_config
from scripts.core.monitoring import MonitoringServer, record_prediction

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
        self._monitor = None

    def run(self):
        cfg = load_config()
        ds_type = cfg.get("data_source", "file")

        logger.info("=" * 50)
        logger.info(f"能源预测引擎 启动 (数据源: {ds_type})")
        logger.info(f"覆盖 {len(get_provinces())} 省, 类型自动扫描")
        logger.info("=" * 50)

        # 监控端点
        try:
            self._monitor = MonitoringServer(port=9090)
            self._monitor.start()
            logger.info("监控端点: http://0.0.0.0:9090")
        except Exception as e:
            logger.warning(f"监控端点启动失败: {e}")

        # 1. 构建特征 (增量 or 全量)
        try:
            self.orch.build_features(force_rebuild=self.force_rebuild_features)
        except Exception as e:
            logger.warning(f"特征构建失败 (将使用已有特征): {e}")

        # 2. 扫描数据 + 全量训练
        try:
            available = self.orch._scan_available()
            if available:
                self.orch.train_all()
                logger.info(f"训练完成: {len(available)} 组")
            else:
                logger.warning("无可用数据, 跳过训练")
                return
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return

        # 3. 预测 + 验证 + 自优化
        self._predict_and_validate()

        # 4. 输出总结
        self._print_summary()

        if self._monitor:
            self._monitor.stop()

    def _predict_and_validate(self):
        for province in get_provinces():
            for target_type in get_available_types(province):
                key = f"{province}_{target_type}"
                try:
                    result = self.orch.predict(province, target_type, 24)
                    health = result.get("health", {})
                    record_prediction(province, target_type,
                                      health.get("mape", 0) or 0,
                                      result.get("n_predictions", 0),
                                      health.get("status", "ok"))

                    # 验证 + 退化检测 → 自动优化
                    val = self.orch.run_validation_cycle(province, target_type)
                    mape = None
                    if val.get("status") == "improved":
                        imp = val.get("improvement", {})
                        logger.info(f"[{key}] 自动优化: "
                                    f"策略={imp.get('selected_strategy')}, "
                                    f"MAPE {imp.get('mape_before')}→{imp.get('mape_after')}")
                    elif "report" in val:
                        mape = val["report"]["metrics"].get("mape")
                        logger.info(f"[{key}] 验证: MAPE={mape}")

                    if mape is not None:
                        self.last_mape[key] = mape

                except Exception as e:
                    logger.error(f"[{key}] 异常: {e}")


    def _print_summary(self):
        logger.info("=" * 50)
        logger.info("本轮总结")
        for key, mape in sorted(self.last_mape.items(), key=lambda x: x[1] or 999):
            logger.info(f"  {key}: MAPE={mape}")
        logger.info("=" * 50)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="能源预测调度引擎 — 单次运行, 配合 cron 使用"
    )
    parser.add_argument("--force-rebuild", action="store_true",
                        help="强制全量重建特征 (代码变更后使用)")
    parser.add_argument("--crontab", action="store_true",
                        help="输出推荐的 crontab 配置")
    args = parser.parse_args()

    d = Daemon()
    d.force_rebuild_features = args.force_rebuild
    d.run()
