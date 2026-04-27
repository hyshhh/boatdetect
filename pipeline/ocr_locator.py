"""
OCRLocator — 基于 PaddleOCR 的弦号文字定位线程

独立于主线程运行，每帧接收图像，使用 OCR 检测船体上的文字区域，
输出文字定位框（虚线框），与主线程的 demo 渲染互不冲突。

功能：
  - 独立线程运行，不阻塞主线程的 YOLO 检测和 Agent 推理
  - 使用 PaddleOCR 的文字检测+识别，直接定位弦号文字
  - 结果通过线程安全队列传递给主线程渲染
  - 支持通过 config 开关控制是否启用
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty

import numpy as np

logger = logging.getLogger(__name__)


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
    OCR 弦号定位器 — 独立线程，使用 PaddleOCR 定位船体文字。

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
        """
        Args:
            enabled: 是否启用 OCR 定位线程。
            use_gpu: 是否使用 GPU 加速 OCR。
            lang: OCR 语言，"ch" 支持中英文混合。
            max_queue_size: 输入/输出队列最大深度。
        """
        self._enabled = enabled
        self._use_gpu = use_gpu
        self._lang = lang
        self._ocr = None

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
            logger.info("OCRLocator 初始化: lang=%s, gpu=%s", lang, use_gpu)
        else:
            logger.info("OCRLocator 已禁用")

    def _init_ocr(self) -> None:
        """延迟初始化 PaddleOCR（首次使用时加载模型）。"""
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
            # PaddleOCR 2.x API — 不受 PaddlePaddle 3.x PIR+OneDNN bug 影响
            # 使用 PP-OCRv4 模型，识别效果好且稳定
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_batch_num=1,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR 模型加载完成")
        except Exception as e:
            logger.error("PaddleOCR 初始化失败: %s", e)
            self._ocr = None

    def start(self) -> None:
        """启动 OCR 定位线程。"""
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
        """停止 OCR 定位线程。"""
        thread = self._thread
        if thread is None:
            return

        self._stop_event.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("OCR 定位线程未在超时内退出（daemon 线程，进程退出时自动回收）")
        self._thread = None

        # 排空队列
        while True:
            try:
                self._input_queue.get_nowait()
            except Empty:
                break
        while True:
            try:
                self._output_queue.get_nowait()
            except Empty:
                break

        logger.info(
            "OCR 定位线程已停止，共处理 %d 帧，平均延迟 %.1fms",
            self._processed_count,
            self._total_latency_ms / max(1, self._processed_count),
        )

    def submit_frame(self, frame: np.ndarray, frame_id: int) -> bool:
        """
        提交一帧给 OCR 线程处理。

        Args:
            frame: BGR 帧图像。
            frame_id: 帧编号。

        Returns:
            True 提交成功，False 队列满或未启用。
        """
        if not self._enabled:
            return False

        # 先检查线程是否存活（避免无意义的入队）
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
        """
        获取最新的 OCR 定位结果（非阻塞或带超时）。

        Returns:
            OCRResult 或 None（无结果时）。
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None

    def drain_results(self) -> list[OCRResult]:
        """排空所有可用的 OCR 结果。"""
        results = []
        while True:
            try:
                results.append(self._output_queue.get_nowait())
            except Empty:
                break
        return results

    def _worker_loop(self) -> None:
        """OCR 工作线程主循环。"""
        self._init_ocr()

        if self._ocr is None:
            logger.error("OCR 未初始化，工作线程退出")
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

            # 放入输出队列（满了就丢弃旧的，保证不阻塞）
            try:
                self._output_queue.put_nowait(result)
            except Exception:
                # 队列满，尝试丢弃最旧的为新结果腾出空间
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    pass
                # 无论丢弃是否成功，都尝试放入新结果
                try:
                    self._output_queue.put_nowait(result)
                except Exception:
                    logger.debug("OCR 输出队列满，丢弃结果 (frame=%d)", frame_id)

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> OCRResult:
        """
        对单帧执行 OCR 文字检测与识别。

        Args:
            frame: BGR 帧。
            frame_id: 帧编号。

        Returns:
            OCRResult 包含检测到的文字框、文字内容、置信度。
        """
        result = OCRResult(frame_id=frame_id, timestamp=time.time())

        try:
            # PaddleOCR 2.x 用 .ocr(cls=True)，3.x 用 .predict()
            try:
                ocr_output = self._ocr.ocr(frame, cls=True)
            except TypeError:
                ocr_output = self._ocr.predict(frame)
        except Exception as e:
            logger.warning("OCR 处理异常 (frame=%d): %s", frame_id, e)
            return result

        if not ocr_output:
            return result

        # 调试：打印 PaddleOCR 输出格式（只打印前 3 帧）
        if frame_id <= 3:
            logger.warning("OCR DEBUG frame=%d type=%s output=%s", frame_id, type(ocr_output).__name__, str(ocr_output)[:500])

        # 兼容 2.x 和 3.x 输出格式
        # 2.x: [[ [box, (text, conf)], ... ]]
        # 3.x dict: [ { "rec_texts": [...], "rec_scores": [...], "dt_polys": [...] } ]
        # 3.x object: [ OCRResult_obj ]
        if isinstance(ocr_output, list) and len(ocr_output) > 0:
            first = ocr_output[0]
        else:
            return result

        # 3.x dict 格式
        if isinstance(first, dict):
            texts = first.get("rec_texts", [])
            scores = first.get("rec_scores", [])
            polys = first.get("dt_polys", [])
            for i in range(min(len(texts), len(polys))):
                text = str(texts[i]).strip()
                conf = float(scores[i]) if i < len(scores) else 0.0
                box = np.array(polys[i], dtype=np.int32)
                result.boxes.append(box)
                result.texts.append(text)
                result.confidences.append(conf)
            return result

        # 2.x list 格式
        lines = first if isinstance(first, list) else []
        for line in lines:
            if line is None or len(line) < 2:
                continue

            box_data = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text_info = line[1]  # (text, confidence)

            if isinstance(text_info, tuple) and len(text_info) >= 2:
                text = str(text_info[0]).strip()
                conf = float(text_info[1])
            else:
                text = str(text_info).strip()
                conf = 0.0

            # 转换为 numpy 数组
            box = np.array(box_data, dtype=np.int32)

            result.boxes.append(box)
            result.texts.append(text)
            result.confidences.append(conf)

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
        """返回统计信息。"""
        avg = self._total_latency_ms / max(1, self._processed_count)
        return {
            "processed_frames": self._processed_count,
            "avg_latency_ms": round(avg, 1),
            "is_running": self.is_running,
        }
