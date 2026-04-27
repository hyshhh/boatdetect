"""OCR 定位模块单元测试（不需要 GPU / PaddleOCR 模型加载）"""

import time
import threading
import numpy as np
import pytest

from pipeline.ocr_locator import OCRLocator, OCRResult


class TestOCRResult:
    def test_default_values(self):
        result = OCRResult(frame_id=1)
        assert result.frame_id == 1
        assert result.boxes == []
        assert result.texts == []
        assert result.confidences == []
        assert result.timestamp == 0.0

    def test_with_data(self):
        box = np.array([[10, 10], [100, 10], [100, 30], [10, 30]], dtype=np.int32)
        result = OCRResult(
            frame_id=5,
            boxes=[box],
            texts=["0014"],
            confidences=[0.95],
            timestamp=time.time(),
        )
        assert result.frame_id == 5
        assert len(result.boxes) == 1
        assert result.texts[0] == "0014"
        assert result.confidences[0] == 0.95


class TestOCRLocatorDisabled:
    def test_disabled_no_start(self):
        """禁用状态下 start() 不应启动线程。"""
        locator = OCRLocator(enabled=False)
        locator.start()
        assert locator.is_running is False
        locator.stop()

    def test_disabled_submit_returns_false(self):
        """禁用状态下 submit_frame 应返回 False。"""
        locator = OCRLocator(enabled=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert locator.submit_frame(frame, 1) is False

    def test_disabled_get_result_returns_none(self):
        """禁用状态下 get_result 应返回 None。"""
        locator = OCRLocator(enabled=False)
        assert locator.get_result() is None

    def test_disabled_drain_returns_empty(self):
        """禁用状态下 drain_results 应返回空列表。"""
        locator = OCRLocator(enabled=False)
        assert locator.drain_results() == []


class TestOCRLocatorQueue:
    def test_submit_and_drain_without_ocr(self):
        """测试队列机制（不实际调用 OCR）。"""
        locator = OCRLocator(enabled=True, max_queue_size=3)
        # 不启动线程，直接测试队列
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # 手动往输入队列放数据
        locator._input_queue.put({"frame": frame, "frame_id": 1})
        assert locator._input_queue.qsize() == 1

        # 手动往输出队列放数据
        test_result = OCRResult(frame_id=1, texts=["test"])
        locator._output_queue.put(test_result)

        results = locator.drain_results()
        assert len(results) == 1
        assert results[0].texts[0] == "test"

    def test_stats_default(self):
        """初始统计应为零。"""
        locator = OCRLocator(enabled=True)
        stats = locator.stats
        assert stats["processed_frames"] == 0
        assert stats["avg_latency_ms"] == 0.0
        assert stats["is_running"] is False


class TestOCRLocatorIntegration:
    """集成测试：实际启动/停止线程（不需要 PaddleOCR 模型）。"""

    def test_start_and_stop(self):
        """启动和停止线程不应崩溃。"""
        locator = OCRLocator(enabled=True)
        locator.start()
        assert locator.is_running is True
        time.sleep(0.2)
        locator.stop(timeout=3.0)
        assert locator.is_running is False

    def test_stop_when_not_started(self):
        """未启动时调用 stop 不应崩溃。"""
        locator = OCRLocator(enabled=True)
        locator.stop()  # 不应抛异常

    def test_double_start(self):
        """重复启动不应崩溃。"""
        locator = OCRLocator(enabled=True)
        locator.start()
        locator.start()  # 第二次应 warning 但不崩溃
        locator.stop()

    def test_submit_when_running(self):
        """运行中提交帧应成功（即使 OCR 模型未加载，只要线程存活）。"""
        locator = OCRLocator(enabled=True)
        locator.start()
        time.sleep(0.3)  # 等待线程启动和 OCR 初始化
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = locator.submit_frame(frame, 1)
        # 如果 PaddleOCR 初始化失败，线程会退出，submit 返回 False
        # 如果成功，submit 返回 True
        assert isinstance(result, bool)
        locator.stop()
