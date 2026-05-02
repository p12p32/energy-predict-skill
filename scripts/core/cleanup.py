"""cleanup.py — 进程清理: signal + atexit 双重保障，防止僵尸进程."""
import os
import sys
import signal
import atexit
import logging
import threading
from typing import Set

logger = logging.getLogger(__name__)

_registered: Set = set()
_is_shutting_down: threading.Event = threading.Event()
_servers: list = []


def register(server_or_resource):
    """注册需要清理的资源."""
    _registered.add(server_or_resource)
    if hasattr(server_or_resource, "stop"):
        _servers.append(server_or_resource)


def shutdown():
    """关闭所有注册的资源."""
    if _is_shutting_down.is_set():
        return
    _is_shutting_down.set()

    for s in _servers:
        try:
            s.stop()
        except Exception:
            pass

    for r in _registered:
        try:
            if hasattr(r, "close"):
                r.close()
            elif hasattr(r, "cleanup"):
                r.cleanup()
        except Exception:
            pass

    _registered.clear()
    _servers.clear()


def _handle_signal(signum, frame):
    logger.info("收到信号 %s，清理退出...", signum)
    shutdown()
    os._exit(0)


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)
atexit.register(shutdown)


def is_shutting_down() -> bool:
    return _is_shutting_down.is_set()
