"""daemon.py — 自动调度闭环引擎

持续运行: 预测 → 验证 → 诊断 → 优化 → 部署 → 下一轮.
支持守护进程模式 和 单次手动模式.
"""
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List

from src.orchestrator import Orchestrator
from src.config_loader import get_provinces, get_types

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
        self.interval = timedelta(minutes=interval_minutes)
        self.running = False

        # 每省/类型 独立的状态
        self.rounds: Dict[str, int] = {}    # 轮次计数
        self.last_predict: Dict[str, datetime] = {}
        self.last_mape: Dict[str, float] = {}

    def start(self, once: bool = False):
        """启动调度循环.

        Args:
            once: True=单次运行后退出, False=持续运行
        """
        self.running = True
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        logger.info("=" * 50)
        logger.info("能源预测引擎 启动")
        logger.info(f"覆盖 {len(get_provinces())} 省 × {len(get_types())} 类型")
        logger.info(f"调度间隔: {self.interval}")
        logger.info("=" * 50)

        # ── 首次初始化 ──
        try:
            self.orch.setup()
            logger.info("表结构检查完成")
        except Exception as e:
            logger.error(f"表初始化失败: {e} (可能 Doris 未连接，尝试继续)")

        # ── 首次训练 ──
        try:
            self.orch.train_all()
            logger.info("首次全量训练完成")
        except Exception as e:
            logger.error(f"训练失败: {e} (可能数据表为空)")

        # ── 主循环 ──
        while self.running:
            cycle_start = datetime.now()
            logger.info("-" * 40)
            logger.info(f"开始新周期 @ {cycle_start.strftime('%H:%M:%S')}")

            for province in get_provinces():
                for target_type in get_types():
                    if not self.running:
                        break
                    self._process(province, target_type)

            if once:
                self._print_summary()
                break

            elapsed = datetime.now() - cycle_start
            sleep_time = max(0, (self.interval - elapsed).total_seconds())
            logger.info(f"周期完成, 等待 {sleep_time:.0f}s 进入下一轮...")
            time.sleep(sleep_time)

    def _process(self, province: str, target_type: str):
        """处理单一省份/类型的一轮完整流程:
        预测 → 验证 → (如果退化) 优化 → 部署
        """
        key = f"{province}_{target_type}"
        self.rounds[key] = self.rounds.get(key, 0) + 1
        round_num = self.rounds[key]

        try:
            # ── Step 1: 预测 ──
            logger.info(f"[{key}#{round_num}] 开始预测...")
            pred_result = self.orch.predict(province, target_type, horizon_hours=24)
            n_pred = pred_result.get("n_predictions", 0)
            logger.info(f"[{key}#{round_num}] 预测完成: {n_pred} 步, "
                        f"样本={pred_result.get('sample', [])}")
            self.last_predict[key] = datetime.now()

            # ── Step 2: 验证 ──
            # 上轮预测值应已在 DB 中, 如果有新真实数据则做对比
            if round_num == 1:
                logger.info(f"[{key}#{round_num}] 首轮, 跳过验证(无历史预测)")
                return

            logger.info(f"[{key}#{round_num}] 执行验证...")
            val_result = self.orch.run_validation_cycle(province, target_type)

            mape = None
            if val_result.get("status") == "no_data":
                logger.info(f"[{key}#{round_num}] 无待验证数据")
                return

            if "validation" in val_result:
                mape = val_result["validation"]["metrics"].get("mape")
            elif "report" in val_result:
                mape = val_result["report"]["metrics"].get("mape")

            if mape is not None:
                self.last_mape[key] = mape

            # ── Step 3: 如果触发改进, 已自动完成优化+重训 ──
            if val_result.get("status") == "improved":
                imp = val_result.get("improvement", {})
                logger.info(
                    f"[{key}#{round_num}] 自动优化完成: "
                    f"策略={imp.get('selected_strategy')}, "
                    f"MAPE {imp.get('mape_before')} → {imp.get('mape_after')}"
                )
            else:
                # 即使未触发, 每 24 轮做一次回塑诊断(温和巡检)
                if round_num % 96 == 0:  # 每日 4 次 × 24 天 = 一次
                    logger.info(f"[{key}#{round_num}] 定期回塑诊断...")
                    bt_result = self.orch.run_backtest_cycle(province, target_type)
                    diag_count = len(bt_result.get("diagnoses", []))
                    logger.info(
                        f"[{key}#{round_num}] 回塑完成: "
                        f"MAPE={bt_result.get('mape')}, 诊断={diag_count}条"
                    )

        except Exception as e:
            logger.error(f"[{key}#{round_num}] 处理异常: {e}", exc_info=True)

    def _print_summary(self):
        logger.info("=" * 50)
        logger.info("本轮总结")
        for key, mape in sorted(self.last_mape.items(), key=lambda x: x[1] or 0):
            logger.info(f"  {key}: MAPE={mape}")
        logger.info("=" * 50)

    def _handle_stop(self, signum, frame):
        logger.info("收到停止信号, 正在安全退出...")
        self.running = False


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="能源预测自动调度引擎")
    parser.add_argument(
        "--interval", type=int, default=15,
        help="调度间隔(分钟), 默认15"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="单次运行后退出"
    )
    args = parser.parse_args()

    d = Daemon(interval_minutes=args.interval)
    d.start(once=args.once)
