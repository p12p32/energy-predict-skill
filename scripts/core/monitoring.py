"""monitoring.py — 健康检查端点 + Prometheus 指标暴露

内置轻量 HTTP server，不引入新依赖。
在 daemon 中作为后台线程运行。

端点:
  GET /health   — 存活检查 (200 OK)
  GET /metrics  — Prometheus 文本格式指标
"""
import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, Optional


# 全局指标存储 (线程安全由 GIL 保证)
_metrics: Dict[str, float] = {}
_labels: Dict[str, str] = {}
_start_time = time.time()


def set_gauge(name: str, value: float, labels: Dict[str, str] = None):
    _metrics[name] = value
    if labels:
        _labels[name] = str(labels)


def inc_counter(name: str, delta: float = 1.0):
    _metrics[name] = _metrics.get(name, 0) + delta


def record_prediction(province: str, target_type: str,
                      mape: float, n_predictions: int, status: str):
    """记录一次预测的结果."""
    key = f"{province}/{target_type}"
    set_gauge("energy_prediction_mape", mape,
              {"province": province, "type": target_type})
    set_gauge("energy_prediction_count", float(n_predictions),
              {"province": province, "type": target_type})
    set_gauge("energy_prediction_status", 1.0 if status == "ok" else 0.0,
              {"province": province, "type": target_type, "key": key})
    inc_counter("energy_predictions_total")
    if status != "ok":
        inc_counter("energy_predictions_degraded")


def get_uptime_seconds() -> float:
    return time.time() - _start_time


def get_metrics() -> str:
    """以 Prometheus 文本格式导出所有指标."""
    lines = [
        f"# HELP energy_uptime_seconds Daemon 运行时间",
        f"# TYPE energy_uptime_seconds gauge",
        f"energy_uptime_seconds {get_uptime_seconds():.1f}",
    ]

    for name, value in sorted(_metrics.items()):
        safe_name = name.replace("-", "_").replace("/", "_")
        label_str = _labels.get(name, "")
        if label_str:
            lines.append(f"# HELP {safe_name} Auto-generated metric")
            lines.append(f"# TYPE {safe_name} gauge")
            lines.append(f"{safe_name}{{{label_str}}} {value}")
        else:
            lines.append(f"# HELP {safe_name} Auto-generated metric")
            lines.append(f"# TYPE {safe_name} gauge")
            lines.append(f"{safe_name} {value}")

    lines.append("")
    return "\n".join(lines)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._health()
        elif self.path == "/metrics":
            self._metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def _health(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        body = json.dumps({
            "status": "ok",
            "uptime_seconds": round(get_uptime_seconds(), 1),
            "predictions_total": int(_metrics.get("energy_predictions_total", 0)),
        })
        self.wfile.write(body.encode())

    def _metrics(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.end_headers()
        self.wfile.write(get_metrics().encode())

    def log_message(self, format, *args):
        pass  # 抑制 HTTP 访问日志


class MonitoringServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._server = HTTPServer((self.host, self.port), HealthHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None
