"""daemon.py — 自动调度引擎 (Watcher 驱动)

持续运行: 数据进来 → 构建特征 → 预测 → 验证 → 退化→优化 → 下一轮.
不再盲睡, 而是等新数据到达才触发流水线.
"""
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

from src.orchestrator import Orchestrator
from src.data_watcher import DataWatcher
from src.core.config import get_provinces, get_types, load_config

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
    def __init__(self, interval_minutes: int = 15):
        self.orch = Orchestrator()
        self.watcher = DataWatcher(self.orch.store.source)
        self.interval = timedelta(minutes=interval_minutes)
        self.running = False
        self.rounds: Dict[str, int] = {}
        self.last_mape: Dict[str, float] = {}
        self.last_predict: Dict[str, datetime] = {}

    def start(self, once: bool = False):
        self.running = True
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        cfg = load_config()
        ds_type = cfg.get("data_source", "file")
        watch_dir = cfg.get("watch_dir", "")

        logger.info("=" * 50)
        logger.info(f"能源预测引擎 启动 (数据源: {ds_type})")
        logger.info(f"覆盖 {len(get_provinces())} 省 × {len(get_types())} 类型")
        logger.info("=" * 50)

        # 首次训练（如果没有已有模型）
        try:
            available = self.orch._scan_available()
            if available:
                self.orch.train_all()
                logger.info(f"首次训练完成: {len(available)} 组")
            else:
                logger.info("无历史数据, 等待新数据...")
        except Exception as e:
            logger.error(f"训练失败: {e}")

        # 主循环
        if once:
            self._single_run(ds_type, watch_dir)
        else:
            self._continuous_loop(ds_type, watch_dir)

    def _single_run(self, ds_type: str, watch_dir: str):
        """单次: 拉取最新数据 → 预测 → 验证."""
        new_batches = self._fetch_new_data(ds_type, watch_dir)

        if new_batches == 0:
            logger.info("无新数据, 只做一次预测")
            for province in get_provinces():
                for target_type in get_types():
                    try:
                        self.orch.predict(province, target_type, 24)
                    except Exception:
                        pass
        else:
            self._run_cycle()

        self._print_summary()

    def _continuous_loop(self, ds_type: str, watch_dir: str):
        """持续: watcher 发现新数据 → 触发完整流水线."""
        while self.running:
            cycle_start = datetime.now()
            logger.info("-" * 40)
            logger.info(f"轮询新数据 @ {cycle_start.strftime('%H:%M:%S')}")

            new_batches = self._fetch_new_data(ds_type, watch_dir)

            if new_batches > 0:
                logger.info(f"发现 {new_batches} 批新数据, 触发流水线...")
                self._run_cycle()
            else:
                logger.info("无新数据, 跳过本轮")

            elapsed = datetime.now() - cycle_start
            sleep_time = max(0, (self.interval - elapsed).total_seconds())
            logger.info(f"等待 {sleep_time:.0f}s 后下一轮轮询...")
            time.sleep(sleep_time)

    def _fetch_new_data(self, ds_type: str, watch_dir: str) -> int:
        """拉取新数据. 返回导入的批次数."""
        if ds_type == "doris":
            return self.watcher.watch_doris(
                callback=lambda p, t, m: self._on_new_data(p, t, m)
            )
        elif watch_dir:
            return self.watcher.watch_file(
                watch_dir,
                callback=lambda p, t, m: self._on_new_data(p, t, m)
            )
        elif ds_type == "file":
            # 默认: 监控 .energy_data/raw/ 目录
            import os
            default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".energy_data", "raw")
            if os.path.isdir(default_dir):
                return self.watcher.watch_file(
                    default_dir,
                    callback=lambda p, t, m: self._on_new_data(p, t, m)
                )
        return 0

    def _on_new_data(self, province: str, target_type: str, msg: str):
        """新数据到达时的回调: 立即触发该 province/type 的预测+验证链."""
        key = f"{province}_{target_type}"
        self.rounds[key] = self.rounds.get(key, 0) + 1
        round_num = self.rounds[key]

        try:
            # 预测
            logger.info(f"[{key}#{round_num}] 新数据到达, 开始预测...")
            pred_result = self.orch.predict(province, target_type, 24)
            n_pred = pred_result.get("n_predictions", 0)
            self.last_predict[key] = datetime.now()
            logger.info(f"[{key}#{round_num}] 预测完成: {n_pred} 步")

            # 如果已有历史预测可对比 → 验证
            if round_num > 1:
                self._validate_and_improve(province, target_type, key, round_num)

        except Exception as e:
            logger.error(f"[{key}#{round_num}] 异常: {e}", exc_info=True)

    def _run_cycle(self):
        """批量: 遍历所有 province/type 跑预测+验证."""
        for province in get_provinces():
            for target_type in get_types():
                if not self.running:
                    break
                key = f"{province}_{target_type}"
                self.rounds[key] = self.rounds.get(key, 0) + 1
                rn = self.rounds[key]

                try:
                    self.orch.predict(province, target_type, 24)
                    if rn > 1:
                        self._validate_and_improve(province, target_type, key, rn)
                except Exception as e:
                    logger.error(f"[{key}#{rn}] 异常: {e}")

    def _validate_and_improve(self, province: str, target_type: str, key: str, rn: int):
        """验证 + 如果退化就触发自主优化."""
        val_result = self.orch.run_validation_cycle(province, target_type)

        mape = None
        if val_result.get("status") == "improved":
            imp = val_result.get("improvement", {})
            logger.info(f"[{key}#{rn}] 自动优化: "
                        f"策略={imp.get('selected_strategy')}, "
                        f"MAPE {imp.get('mape_before')}→{imp.get('mape_after')}")
        elif "report" in val_result:
            mape = val_result["report"]["metrics"].get("mape")
            logger.info(f"[{key}#{rn}] 验证: MAPE={mape}")

        if mape is not None:
            self.last_mape[key] = mape

    def _print_summary(self):
        logger.info("=" * 50)
        logger.info("本轮总结")
        for key, mape in sorted(self.last_mape.items(), key=lambda x: x[1] or 0):
            logger.info(f"  {key}: MAPE={mape}")
        logger.info("=" * 50)

    def _handle_stop(self, signum, frame):
        logger.info("收到停止信号, 安全退出...")
        self.running = False


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="能源预测自动调度引擎")
    parser.add_argument("--interval", type=int, default=15, help="轮询间隔(分钟)")
    parser.add_argument("--once", action="store_true", help="单次运行后退出")
    parser.add_argument("--source", type=str, default=None, help="数据源: file|doris|api")
    parser.add_argument("--watch", type=str, default=None, help="监控的目录 (file 模式)")
    args = parser.parse_args()

    if args.source:
        import os
        os.environ["DATA_SOURCE"] = args.source

    d = Daemon(interval_minutes=args.interval)
    d.start(once=args.once)
