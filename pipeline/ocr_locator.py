"""
OCRLocator — 纯 OpenCV 文字区域定位线程

独立于主线程运行，每帧接收图像，用形态学 + 轮廓检测定位船体上的文字区域，
输出定位框（虚线框），与主线程的 demo 渲染互不冲突。

无任何 OCR 引擎依赖，仅使用 OpenCV。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """单帧的文字定位结果（仅位置，无识别文本）。"""
    frame_id: int
    boxes: list[np.ndarray] = field(default_factory=list)   # 每个 box 是 4x2 的 ndarray
    timestamp: float = 0.0


class OCRLocator:
    """
    文字区域定位器 — 独立线程，纯 OpenCV 实现。

    工作流程：
    1. 主线程将 frame 送入输入队列
    2. 定位线程从队列取帧，检测文字区域轮廓
    3. 结果（bounding boxes）放入输出队列
    4. 主线程从输出队列取结果，绘制虚线框
    """

    def __init__(
        self,
        enabled: bool = True,
        use_gpu: bool = False,
        lang: str = "ch",
        max_queue_size: int = 5,
    ):
        self._enabled = enabled

        # 输入队列：主线程 → 定位线程
        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        # 输出队列：定位线程 → 主线程
        self._output_queue: Queue = Queue(maxsize=max_queue_size)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # 统计
        self._processed_count = 0
        self._total_latency_ms = 0.0

        if enabled:
            logger.info("OCRLocator 初始化: 纯 OpenCV 文字定位")
        else:
            logger.info("OCRLocator 已禁用")

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            logger.warning("OCR 定位线程已在运行")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="ocr-locator",
            daemon=True,
        )
        self._thread.start()
        logger.info("OCR 定位线程已启动")

    def stop(self, timeout: float = 5.0) -> None:
        thread = self._thread
        if thread is None:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        self._thread = None
        for q in (self._input_queue, self._output_queue):
            while True:
                try:
                    q.get_nowait()
                except Empty:
                    break
        logger.info(
            "OCR 定位线程已停止，共处理 %d 帧，平均延迟 %.1fms",
            self._processed_count,
            self._total_latency_ms / max(1, self._processed_count),
        )

    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        if not self._enabled:
            return False
        thread = self._thread
        if thread is None or not thread.is_alive():
            return False
        try:
            self._input_queue.put_nowait({"frame": frame.copy(), "frame_id": frame_id})
            return True
        except Exception:
            return False

    def get_result(self, timeout: float = 0.0) -> OCRResult | None:
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None

    def drain_results(self) -> list[OCRResult]:
        results = []
        while True:
            try:
                results.append(self._output_queue.get_nowait())
            except Empty:
                break
        return results

    # ── 内部 ──────────────────────────────────────

    def _worker_loop(self) -> None:
        logger.info("OCR 定位线程开始处理")
        while not self._stop_event.is_set():
            try:
                task = self._input_queue.get(timeout=0.5)
            except Empty:
                continue

            frame = task["frame"]
            frame_id = task["frame_id"]

            t0 = time.perf_counter()
            result = self._detect_text_regions(frame, frame_id)
            latency_ms = (time.perf_counter() - t0) * 1000

            self._processed_count += 1
            self._total_latency_ms += latency_ms

            try:
                self._output_queue.put_nowait(result)
            except Exception:
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    pass
                try:
                    self._output_queue.put_nowait(result)
                except Exception:
                    pass

    def _detect_text_regions(self, frame: np.ndarray, frame_id: int) -> OCRResult:
        """
        纯 OpenCV 文字区域检测：
        灰度 → 自适应二值化 → 形态学膨胀 → 轮廓 → 过滤
        """
        result = OCRResult(frame_id=frame_id, timestamp=time.time())
        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 自适应二值化，突出文字笔画
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=15, C=10,
            )

            # 水平方向膨胀，把同一行的文字连成条状
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
            dilated = cv2.dilate(binary, kernel_h, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h

                # 面积过滤：太小的噪点、太大的整块区域
                if area < frame_area * 0.0002 or area > frame_area * 0.12:
                    continue
                # 宽高比：文字条通常是横向的
                if w < h * 0.8:
                    continue
                # 尺寸：宽至少 30px，高至少 8px
                if w < 30 or h < 8:
                    continue

                # 稍微扩大一点，让框更完整
                pad = 4
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_frame, x + w + pad)
                y2 = min(h_frame, y + h + pad)

                box = np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.int32)
                result.boxes.append(box)

        except Exception as e:
            logger.warning("文字定位异常 (frame=%d): %s", frame_id, e)

        if result.boxes:
            logger.debug("文字定位 (frame=%d): %d 个区域", frame_id, len(result.boxes))

        return result

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def stats(self) -> dict:
        avg = self._total_latency_ms / max(1, self._processed_count)
        return {
            "processed_frames": self._processed_count,
            "avg_latency_ms": round(avg, 1),
            "is_running": self.is_running,
        }
