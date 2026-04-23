"""
FPSMeter — 10秒滑动窗口 FPS 统计
LatencyMeter — 10秒滑动窗口阶段耗时统计（avg / p50 / p95 / max）

在终端和日志中定期打印码流帧率和处理帧率，以及各阶段延迟。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class FPSMeter:
    """
    基于滑动窗口的 FPS 计算器。

    支持多个独立计数通道（如 "stream" 和 "process"）。
    """

    def __init__(self, window_seconds: float = 10.0):
        """
        Args:
            window_seconds: 滑动窗口时长（秒）。
        """
        self._window = max(1.0, window_seconds)  # 最小 1 秒
        self._timestamps: dict[str, deque[float]] = {}
        self._last_print: dict[str, float] = {}
        self._print_interval = 5.0  # 每 5 秒打印一次

    def tick(self, channel: str = "default") -> None:
        """记录一次帧到达。"""
        now = time.monotonic()

        if channel not in self._timestamps:
            self._timestamps[channel] = deque()
            self._last_print[channel] = 0.0

        self._timestamps[channel].append(now)

        # 清理窗口外的旧数据
        cutoff = now - self._window
        while self._timestamps[channel] and self._timestamps[channel][0] < cutoff:
            self._timestamps[channel].popleft()

    def get_fps(self, channel: str = "default") -> float:
        """获取当前 FPS。"""
        if channel not in self._timestamps:
            return 0.0

        timestamps = self._timestamps[channel]
        if len(timestamps) < 2:
            return 0.0

        now = time.monotonic()
        cutoff = now - self._window
        # 清理旧数据
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    def should_print(self, channel: str = "default") -> bool:
        """判断是否应该打印 FPS（基于打印间隔）。"""
        now = time.monotonic()
        if channel not in self._last_print:
            self._last_print[channel] = now
            return False

        if now - self._last_print[channel] >= self._print_interval:
            self._last_print[channel] = now
            return True
        return False

    def print_fps(self, channel: str = "default", extra: str = "") -> str:
        """
        格式化打印 FPS 信息。

        Returns:
            格式化后的 FPS 字符串。
        """
        fps = self.get_fps(channel)
        parts = [f"[{channel}] FPS: {fps:.1f}"]
        if extra:
            parts.append(extra)
        msg = " | ".join(parts)
        logger.info(msg)
        return msg

    def get_all_fps(self) -> dict[str, float]:
        """获取所有通道的 FPS。"""
        return {ch: self.get_fps(ch) for ch in self._timestamps}

    def reset(self, channel: str | None = None) -> None:
        """重置计数器。"""
        if channel:
            self._timestamps.pop(channel, None)
            self._last_print.pop(channel, None)
        else:
            self._timestamps.clear()
            self._last_print.clear()


# ── LatencyMeter ───────────────────────────────────────


class LatencyMeter:
    """
    滑动窗口阶段耗时统计。

    支持多个阶段通道（如 "yolo"、"agent"、"demo"），
    每次 record() 记录一次耗时（毫秒），窗口内自动清理过期样本。

    用法：
        meter = LatencyMeter(window_seconds=10.0)

        # 手动计时
        t0 = time.perf_counter()
        ... do work ...
        meter.record("yolo", (time.perf_counter() - t0) * 1000)

        # 或用 context manager
        with meter.measure("agent"):
            ... do work ...

        # 查询统计
        stats = meter.get_stats("yolo")
        # {"avg": 35.2, "p50": 33.1, "p95": 48.7, "max": 52.3, "count": 120}
    """

    def __init__(self, window_seconds: float = 10.0):
        self._window = max(1.0, window_seconds)
        # channel → deque of (timestamp, latency_ms)
        self._samples: dict[str, deque[tuple[float, float]]] = {}

    def record(self, channel: str, latency_ms: float) -> None:
        """记录一次耗时（毫秒）。"""
        now = time.monotonic()
        if channel not in self._samples:
            self._samples[channel] = deque()
        self._samples[channel].append((now, latency_ms))
        self._cleanup(channel, now)

    @contextmanager
    def measure(self, channel: str):
        """上下文管理器，自动计时并记录。"""
        t0 = time.perf_counter()
        yield
        latency_ms = (time.perf_counter() - t0) * 1000
        self.record(channel, latency_ms)

    def _cleanup(self, channel: str, now: float) -> None:
        """清理窗口外的旧样本。"""
        cutoff = now - self._window
        samples = self._samples[channel]
        while samples and samples[0][0] < cutoff:
            samples.popleft()

    def get_stats(self, channel: str) -> dict[str, float]:
        """
        获取通道的延迟统计（单位 ms）。

        返回 {"avg", "p50", "p95", "max", "count"}。
        无样本时全为 0。
        """
        if channel not in self._samples:
            return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "count": 0}

        # 先清理
        now = time.monotonic()
        self._cleanup(channel, now)

        samples = self._samples[channel]
        if not samples:
            return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "count": 0}

        values = sorted(s[1] for s in samples)
        count = len(values)
        total = sum(values)

        avg = total / count
        p50 = values[count // 2]
        p95_idx = min(int(count * 0.95), count - 1)
        p95 = values[p95_idx]
        max_val = values[-1]

        return {
            "avg": round(avg, 1),
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "max": round(max_val, 1),
            "count": count,
        }

    def get_all_stats(self) -> dict[str, dict[str, float]]:
        """获取所有通道的统计。"""
        return {ch: self.get_stats(ch) for ch in self._samples}

    def reset(self, channel: str | None = None) -> None:
        """重置。"""
        if channel:
            self._samples.pop(channel, None)
        else:
            self._samples.clear()
