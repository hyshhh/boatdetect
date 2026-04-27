"""
OCRLocator — 纯 OpenCV 文字区域定位线程

独立于主线程运行，每帧接收图像，用形态学 + 轮廓检测定位船体上的文字区域，
输出定位框（虚线框），与主线程的 demo 渲染互不冲突。
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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算两个 (x, y, w, h) 矩形的 IoU。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_rects(
    rects: list[tuple[int, int, int, int]],
    iou_thresh: float = 0.3,
) -> list[int]:
    """对 (x, y, w, h) 列表做 NMS，返回保留的索引。"""
    if not rects:
        return []
    areas = [r[2] * r[3] for r in rects]
    order = sorted(range(len(rects)), key=lambda i: areas[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        remaining = []
        for j in order:
            if _iou(rects[i], rects[j]) <= iou_thresh:
                remaining.append(j)
        order = remaining
    return keep


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
        max_queue_size: int = 5,
        nms_iou_threshold: float = 0.3,
        target_width: int = 640,
        min_area_ratio: float = 0.0004,
        max_area_ratio: float = 0.1,
        min_aspect_ratio: float = 1.5,
        min_edge_density: float = 0.08,
        min_gray_std: float = 15.0,
    ):
        self._enabled = enabled
        self._nms_iou_thresh = nms_iou_threshold
        self._target_width = target_width
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        self._min_aspect_ratio = min_aspect_ratio
        self._min_edge_density = min_edge_density
        self._min_gray_std = min_gray_std

        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        self._output_queue: Queue = Queue(maxsize=max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._processed_count = 0
        self._total_latency_ms = 0.0

        if enabled:
            logger.info("OCRLocator 初始化: nms_iou=%.2f, target_w=%d", nms_iou_threshold, target_width)
        else:
            logger.info("OCRLocator 已禁用")

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="ocr-locator", daemon=True)
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
        avg = self._total_latency_ms / max(1, self._processed_count)
        logger.info("OCR 定位线程已停止，共处理 %d 帧，平均 %.1fms", self._processed_count, avg)

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
        while not self._stop_event.is_set():
            try:
                task = self._input_queue.get(timeout=0.5)
            except Empty:
                continue

            t0 = time.perf_counter()
            result = self._detect_text_regions(task["frame"], task["frame_id"])
            self._total_latency_ms += (time.perf_counter() - t0) * 1000
            self._processed_count += 1

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
        """缩小 → 边缘检测 → 水平膨胀 → 轮廓 → 多重过滤 → NMS"""
        result = OCRResult(frame_id=frame_id, timestamp=time.time())
        h_orig, w_orig = frame.shape[:2]

        try:
            # 缩小
            scale = self._target_width / w_orig
            small = cv2.resize(frame, (self._target_width, int(h_orig * scale)))
            h, w = small.shape[:2]
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 80, 200)

            # 水平膨胀，把同行文字边缘连成条
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 2))
            dilated = cv2.dilate(edges, kernel_h, iterations=1)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 100:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:100]

            frame_area = h * w
            candidates: list[tuple[int, int, int, int]] = []
            candidate_boxes: list[np.ndarray] = []

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch

                if area < frame_area * self._min_area_ratio or area > frame_area * self._max_area_ratio:
                    continue
                if cw < ch * self._min_aspect_ratio:
                    continue
                if cw < 25 or ch < 6:
                    continue

                roi_edges = edges[y:y+ch, x:x+cw]
                if roi_edges.size == 0:
                    continue
                if np.count_nonzero(roi_edges) / roi_edges.size < self._min_edge_density:
                    continue

                roi_gray = gray[y:y+ch, x:x+cw]
                if roi_gray.size == 0:
                    continue
                if np.std(roi_gray) < self._min_gray_std:
                    continue

                # 映射回原图坐标
                pad = int(3 / scale)
                x1 = max(0, int(x / scale) - pad)
                y1 = max(0, int(y / scale) - pad)
                x2 = min(w_orig, int((x + cw) / scale) + pad)
                y2 = min(h_orig, int((y + ch) / scale) + pad)

                candidates.append((x1, y1, x2 - x1, y2 - y1))
                candidate_boxes.append(
                    np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                )

            # NMS 去重
            keep_indices = _nms_rects(candidates, self._nms_iou_thresh)
            result.boxes = [candidate_boxes[i] for i in keep_indices]

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
