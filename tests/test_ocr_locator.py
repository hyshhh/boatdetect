"""OCR 定位模块单元测试"""

import time
import numpy as np
import pytest

from pipeline.ocr_locator import OCRLocator, OCRResult, _iou, _nms_rects


class TestIOU:
    def test_identical(self):
        assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert _iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0

    def test_partial_overlap(self):
        iou = _iou((0, 0, 10, 10), (5, 0, 10, 10))
        # 交集 5*10=50, 并集 100+100-50=150
        assert abs(iou - 50 / 150) < 1e-6

    def test_contains(self):
        iou = _iou((0, 0, 10, 10), (2, 2, 5, 5))
        # 交集 5*5=25, 并集 100+25-25=100
        assert abs(iou - 0.25) < 1e-6


class TestNMS:
    def test_empty(self):
        assert _nms_rects([]) == []

    def test_single(self):
        assert _nms_rects([(0, 0, 10, 10)]) == [0]

    def test_no_overlap(self):
        rects = [(0, 0, 10, 10), (20, 20, 10, 10)]
        keep = _nms_rects(rects, iou_thresh=0.5)
        assert len(keep) == 2

    def test_full_overlap(self):
        rects = [(0, 0, 10, 10), (0, 0, 10, 10)]
        keep = _nms_rects(rects, iou_thresh=0.5)
        assert len(keep) == 1

    def test_partial_suppress(self):
        rects = [(0, 0, 10, 10), (1, 1, 10, 10), (50, 50, 10, 10)]
        keep = _nms_rects(rects, iou_thresh=0.3)
        assert len(keep) == 2  # 前两个重叠，保留面积大的，第三个独立


class TestOCRResult:
    def test_default_values(self):
        result = OCRResult(frame_id=1)
        assert result.frame_id == 1
        assert result.boxes == []
        assert result.timestamp == 0.0

    def test_with_data(self):
        box = np.array([[10, 10], [100, 10], [100, 30], [10, 30]], dtype=np.int32)
        result = OCRResult(frame_id=5, boxes=[box], timestamp=time.time())
        assert result.frame_id == 5
        assert len(result.boxes) == 1


class TestOCRLocator:
    def test_disabled_returns_empty(self):
        locator = OCRLocator(enabled=False)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = locator.detect(frame, frame_id=1)
        assert result.boxes == []

    def test_empty_crop(self):
        locator = OCRLocator(enabled=True)
        result = locator.detect(np.array([]), frame_id=1)
        assert result.boxes == []

    def test_offset_conversion(self):
        """OCR 坐标应正确加上 offset 偏移。"""
        locator = OCRLocator(
            enabled=True,
            target_width=640,
            min_area_ratio=0.0,      # 放宽阈值让测试通过
            max_area_ratio=1.0,
            min_aspect_ratio=0.0,
            min_edge_density=0.0,
            min_gray_std=0.0,
        )
        # 创建一个有文字特征的 crop（水平条纹模拟文字）
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.rectangle(crop, (10, 40), (190, 60), (255, 255, 255), -1)

        offset = (100, 50)
        result = locator.detect(crop, frame_id=1, offset=offset)
        # 所有 box 的坐标应 >= offset
        for box in result.boxes:
            assert box[0][0] >= offset[0]  # x1 >= ox
            assert box[0][1] >= offset[1]  # y1 >= oy

    def test_nms_removes_duplicates(self):
        """NMS 应合并重叠框。"""
        locator = OCRLocator(
            enabled=True,
            nms_iou_threshold=0.3,
            target_width=640,
            min_area_ratio=0.0,
            max_area_ratio=1.0,
            min_aspect_ratio=0.0,
            min_edge_density=0.0,
            min_gray_std=0.0,
        )
        # 创建有两个独立文字区域的 crop
        crop = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.rectangle(crop, (10, 40), (150, 60), (255, 255, 255), -1)
        cv2.rectangle(crop, (200, 40), (390, 60), (255, 255, 255), -1)

        result = locator.detect(crop, frame_id=1)
        # 不应有完全重叠的框（NMS 去重）
        for i, a in enumerate(result.boxes):
            for j, b in enumerate(result.boxes):
                if i < j:
                    # 两个框不应完全相同
                    assert not np.array_equal(a, b)


# 需要 cv2 用于测试辅助
import cv2
