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
from scripts.core.config import get_provinces, get_available_types, load_config, get_data_delay, get_available_date
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

        # 2. 扫描数据 + 全量训练 (分层架构)
        try:
            result = self.orch.train_all_layered()
            if result.get("status") == "no_data":
                logger.warning("无可用数据, 跳过训练")
                return
            trained = len(result.get("results", {}))
            logger.info(f"分层训练完成: {trained} 组")
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return

        # 3. 预测 + 输出
        self._predict_and_validate()

        # 4. 输出总结
        self._print_summary()

        if self._monitor:
            self._monitor.stop()

    def _check_data_ready(self, province: str, target_type: str) -> bool:
        """检查该 province/type 的数据是否已就绪 (基于 data_availability 延迟配置)."""
        delay = get_data_delay(province, target_type)
        avail_date = get_available_date(province, target_type)

        # 延迟 <= 0 表示提前可用 (如日前预测), 始终就绪
        if delay <= 0:
            return True

        # 延迟 > 0: 检查实际最新数据是否已到达 avail_date
        latest = self.orch._find_latest_date(province, target_type)
        if latest is None:
            logger.warning("[数据就绪] %s/%s: 无已有数据, 延迟=%+dd, 期望截止=%s",
                          province, target_type, delay,
                          avail_date.strftime("%Y-%m-%d"))
            return False

        ready = latest >= avail_date
        if not ready:
            logger.info("[数据就绪] %s/%s: 未就绪 (最新=%s, 期望>=%s, 延迟=%+dd), 跳过",
                       province, target_type,
                       latest.strftime("%Y-%m-%d"),
                       avail_date.strftime("%Y-%m-%d"),
                       delay)
        return ready

    def _predict_and_validate(self):
        skipped = 0
        for province in get_provinces():
            for target_type in get_available_types(province):
                key = f"{province}_{target_type}"

                # 数据就绪检查
                if not self._check_data_ready(province, target_type):
                    skipped += 1
                    continue

                try:
                    df = self.orch.predict_layered(province, target_type, 24)
                    n = len(df)
                    record_prediction(province, target_type, 0, n, "ok")
                    logger.info(f"[{key}] 预测完成: {n} 步")

                except Exception as e:
                    logger.error(f"[{key}] 异常: {e}")

        if skipped > 0:
            logger.info("数据就绪检查: %d 个类型因数据未到库跳过", skipped)


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
