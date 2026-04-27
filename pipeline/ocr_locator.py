"""
OCRLocator — 基于 Tesseract OCR 的弦号文字定位线程

独立于主线程运行，每帧接收图像，使用 OCR 检测船体上的文字区域，
输出文字定位框（虚线框），与主线程的 demo 渲染互不冲突。

功能：
  - 独立线程运行，不阻塞主线程的 YOLO 检测和 Agent 推理
  - 使用 Tesseract OCR 直接定位弦号文字
  - 结果通过线程安全队列传递给主线程渲染
  - 支持通过 config 开关控制是否启用

依赖：
  - 系统：sudo apt install tesseract-ocr tesseract-ocr-chi-sim
  - Python：pip install pytesseract
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

# Tesseract 语言映射
_LANG_MAP = {
    "ch": "chi_sim+eng",
    "en": "eng",
    "chi_sim": "chi_sim+eng",
}


@dataclass
class OCRResult:
    """单帧的 OCR 定位结果。"""
    frame_id: int
    boxes: list[np.ndarray] = field(default_factory=list)   # 每个 box 是 4x2 的 ndarray
    texts: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    timestamp: float = 0.0


class OCRLocator:
    """
    OCR 弦号定位器 — 独立线程，使用 Tesseract OCR 定位船体文字。

    工作流程：
    1. 主线程将 frame 送入输入队列
    2. OCR 线程从队列取帧，执行文字检测+识别
    3. 结果放入输出队列
    4. 主线程从输出队列取结果，绘制虚线框

    线程安全，支持随时启停。
    """

    def __init__(
        self,
        enabled: bool = True,
        use_gpu: bool = False,
        lang: str = "ch",
        max_queue_size: int = 5,
    ):
        self._enabled = enabled
        self._use_gpu = use_gpu
        self._lang = _LANG_MAP.get(lang, lang)
        self._tesseract_ok = False

        # 输入队列：主线程 → OCR 线程
        self._input_queue: Queue = Queue(maxsize=max_queue_size)
        # 输出队列：OCR 线程 → 主线程
        self._output_queue: Queue = Queue(maxsize=max_queue_size)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # 统计
        self._processed_count = 0
        self._total_latency_ms = 0.0

        if enabled:
            logger.info("OCRLocator 初始化: lang=%s (tesseract=%s), gpu=%s", lang, self._lang, use_gpu)
        else:
            logger.info("OCRLocator 已禁用")

    def _init_ocr(self) -> None:
        """延迟初始化 Tesseract OCR。"""
        if self._tesseract_ok:
            return
        try:
            import pytesseract
            # 验证 Tesseract 可用
            pytesseract.get_tesseract_version()
            self._tesseract_ok = True
            logger.info("Tesseract OCR 初始化成功 (lang=%s)", self._lang)
        except ImportError:
            logger.error(
                "pytesseract 未安装。请执行: pip install pytesseract\n"
                "并安装系统包: sudo apt install tesseract-ocr tesseract-ocr-chi-sim"
            )
        except Exception as e:
            logger.error("Tesseract OCR 初始化失败: %s\n请确认已安装: sudo apt install tesseract-ocr tesseract-ocr-chi-sim", e)

    def start(self) -> None:
        if not self._enabled:
            logger.debug("OCR 定位线程未启用，跳过启动")
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
        if thread.is_alive():
            logger.warning("OCR 定位线程未在超时内退出")
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
            self._input_queue.put_nowait({
                "frame": frame.copy(),
                "frame_id": frame_id,
            })
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

    def _worker_loop(self) -> None:
        self._init_ocr()

        if not self._tesseract_ok:
            logger.error("Tesseract OCR 未就绪，工作线程退出")
            return

        logger.info("OCR 工作线程开始处理")

        while not self._stop_event.is_set():
            try:
                task = self._input_queue.get(timeout=0.5)
            except Empty:
                continue

            frame = task["frame"]
            frame_id = task["frame_id"]

            t0 = time.perf_counter()
            result = self._process_frame(frame, frame_id)
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
                    logger.debug("OCR 输出队列满，丢弃结果 (frame=%d)", frame_id)

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> OCRResult:
        """
        对单帧执行 OCR：先用 OpenCV 检测文字区域，再用 Tesseract 识别。
        """
        import pytesseract

        result = OCRResult(frame_id=frame_id, timestamp=time.time())

        try:
            # 1. 预处理：灰度 → 自适应二值化 → 形态学
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
            )
            # 膨胀文字区域使其连通
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)

            # 2. 查找文字区域轮廓
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            h_frame, w_frame = frame.shape[:2]
            min_area = (h_frame * w_frame) * 0.0003   # 最小面积阈值
            max_area = (h_frame * w_frame) * 0.15     # 最大面积阈值

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h

                # 过滤：太小或太大的区域
                if area < min_area or area > max_area:
                    continue
                # 过滤：宽高比不合理（文字区域通常宽 > 高）
                if w < h * 0.5:
                    continue

                # 裁剪区域（带少量 padding）
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_frame, x + w + pad)
                y2 = min(h_frame, y + h + pad)
                roi = frame[y1:y2, x1:x2]

                if roi.size == 0:
                    continue

                # 3. Tesseract 识别
                try:
                    text = pytesseract.image_to_string(
                        roi, lang=self._lang, config="--psm 7 --oem 3"
                    ).strip()

                    # 过滤：空结果或太短
                    if not text or len(text) < 2:
                        continue

                    # 获取置信度
                    try:
                        data = pytesseract.image_to_data(
                            roi, lang=self._lang, config="--psm 7 --oem 3",
                            output_type=pytesseract.Output.DICT
                        )
                        confs = [int(c) for c in data["conf"] if int(c) > 0]
                        conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
                    except Exception:
                        conf = 0.0

                    # 构造四点框
                    box = np.array([
                        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                    ], dtype=np.int32)

                    result.boxes.append(box)
                    result.texts.append(text)
                    result.confidences.append(conf)

                except Exception:
                    continue

        except Exception as e:
            logger.warning("OCR 处理异常 (frame=%d): %s", frame_id, e)

        if result.boxes:
            logger.debug(
                "OCR 定位 (frame=%d): 检测到 %d 个文字区域",
                frame_id, len(result.boxes),
            )

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
