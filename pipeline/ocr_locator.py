"""
OCRLocator — 纯 OpenCV 文字区域定位（同步，无独立线程）

接收 YOLO crop 图像，用形态学 + 轮廓检测定位文字区域，
返回的 box 坐标已转换到原图（frame）坐标系。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """单次文字定位结果（坐标已转换到 frame 坐标系）。"""
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
    union = aw * ah + bw * bh - inter
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
    文字区域定位器 — 纯 OpenCV，同步调用。

    用法：
        locator = OCRLocator(nms_iou_threshold=0.3)
        result = locator.detect(crop, frame_id=1, offset=(x1, y1))
        # result.boxes 已经是 frame 坐标系
    """

    def __init__(
        self,
        enabled: bool = True,
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

    def detect(
        self,
        crop: np.ndarray,
        frame_id: int = 0,
        offset: tuple[int, int] = (0, 0),
    ) -> OCRResult:
        """
        对 crop 图像做文字区域检测，返回 frame 坐标系下的结果。

        Args:
            crop: YOLO 裁剪出的船体图像（BGR）。
            frame_id: 帧号。
            offset: crop 在原图中的左上角偏移 (x_offset, y_offset)。

        Returns:
            OCRResult，boxes 已转换到 frame 坐标系。
        """
        if not self._enabled or crop is None or crop.size == 0:
            return OCRResult(frame_id=frame_id)

        import time
        result = OCRResult(frame_id=frame_id, timestamp=time.time())
        ox, oy = offset
        h_orig, w_orig = crop.shape[:2]

        try:
            # 缩小到 target_width
            scale = self._target_width / w_orig
            small = cv2.resize(crop, (self._target_width, int(h_orig * scale)))
            h, w = small.shape[:2]
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 80, 200)

            # 水平膨胀
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

                # 映射回 crop 坐标，再加 offset 转到 frame 坐标
                pad = int(3 / scale)
                x1 = max(0, int(x / scale) - pad) + ox
                y1 = max(0, int(y / scale) - pad) + oy
                x2 = int((x + cw) / scale) + pad + ox
                y2 = int((y + ch) / scale) + pad + oy

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
