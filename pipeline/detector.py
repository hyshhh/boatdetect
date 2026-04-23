"""
ShipDetector — 基于 YOLO 的船只检测与跟踪

使用 ultralytics YOLO 原生追踪算法（ByteTrack），输出带 track ID 的检测框。
支持从 config 传入自定义 tracker 参数，自动生成 tracker YAML。
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果。"""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray | None = None   # 裁剪的图像区域


def _build_tracker_yaml(
    tracker_type: str,
    tracker_params: dict[str, Any] | None,
) -> str:
    """
    根据 tracker_type 和自定义参数生成 tracker YAML。

    如果 tracker_params 为 None 或空，直接返回 "{tracker_type}.yaml"
    让 ultralytics 使用内置默认配置。

    否则生成临时 YAML 文件并返回其路径。
    """
    if not tracker_params:
        return f"{tracker_type}.yaml"

    # 构建配置字典
    cfg: dict[str, Any] = {"tracker_type": tracker_type}
    cfg.update(tracker_params)

    # 写入临时文件
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix=f"{tracker_type}_", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
    tmp.close()

    logger.info("生成 tracker 配置: %s (type=%s, params=%s)", tmp.name, tracker_type, list(tracker_params.keys()))
    return tmp.name


class ShipDetector:
    """
    YOLO 船只检测器（带原生跟踪）。

    使用 ultralytics YOLO 内置的 ByteTrack 追踪算法，
    为每个检测目标分配持久 track ID。
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "",
        conf_threshold: float = 0.25,
        tracker_type: str = "bytetrack",
        tracker_params: dict[str, Any] | None = None,
        classes: list[int] | None = None,
    ):
        """
        Args:
            model_path: YOLO 模型路径或名称。若不存在会自动下载。
            device: 推理设备，"" 自动选择，"cpu" 强制 CPU，"0" 表示 GPU 0。
            conf_threshold: 检测置信度阈值。
            tracker_type: 追踪算法，"bytetrack" 或 "botsort"。
            tracker_params: 追踪器参数字典。None 使用 ultralytics 内置默认值。
            classes: 只检测指定类别 ID 列表。None 表示检测所有类别。
                     COCO 中船的类别 ID 是 8。
        """
        from ultralytics import YOLO

        self._conf_threshold = conf_threshold
        self._classes = classes
        self._device = device

        # 生成 tracker YAML
        self._tracker_yaml = _build_tracker_yaml(tracker_type, tracker_params)
        self._tracker_type = tracker_type
        self._tracker_tmp_file: str | None = (
            self._tracker_yaml if self._tracker_yaml != f"{tracker_type}.yaml" else None
        )

        logger.info("加载 YOLO 模型: %s (device=%s)", model_path, device or "auto")
        self._model = YOLO(model_path)

        # 兼容性修复：部分 ultralytics 版本 default.yaml 缺少 fuse_score 字段
        # 动态修补 ultralytics 内部配置类，注入缺失的属性
        self._patch_ultralytics_cfg()

        # 预热（忽略结果）
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.track(
                source=dummy,
                persist=True,
                tracker=self._tracker_yaml,
                verbose=False,
                device=device or None,
            )
        except Exception as e:
            logger.warning("YOLO 预热失败（不影响后续使用）: %s", e)

        logger.info("YOLO 模型加载完成，追踪器: %s (配置: %s)", tracker_type, self._tracker_yaml)

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> list[Detection]:
        """
        对单帧图像执行检测 + 跟踪。

        Args:
            frame: BGR 格式的图像帧 (H, W, 3)。
            frame_id: 帧编号（用于日志）。

        Returns:
            检测结果列表，每个包含 track_id, bbox, confidence, crop。
        """
        try:
            results = self._model.track(
                source=frame,
                persist=True,
                conf=self._conf_threshold,
                tracker=self._tracker_yaml,
                classes=self._classes,
                verbose=False,
                device=self._device or None,
            )
        except Exception as e:
            logger.error("YOLO 检测异常 (frame=%d): %s", frame_id, e)
            return []

        detections: list[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes

        # 跟踪器未分配 ID 时，boxes.id 为 None
        if boxes.id is None:
            return detections

        for i in range(len(boxes)):
            # 获取 track ID
            track_id = int(boxes.id[i].item())

            # 获取 bbox (xyxy 格式)
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # 确保 bbox 有效（x2 > x1, y2 > y1）
            if x2 <= x1 or y2 <= y1:
                continue

            # 获取置信度
            conf = float(boxes.conf[i].item())

            # 裁剪图像区域（加大 padding，确保船体文字可见）
            h, w = frame.shape[:2]
            pad = 20
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(w, x2 + pad)
            cy2 = min(h, y2 + pad)
            crop = frame[cy1:cy2, cx1:cx2].copy()

            # 跳过太小的 crop（<80px 侧文字基本看不清）
            crop_h, crop_w = crop.shape[:2]
            if crop_w < 80 or crop_h < 80:
                logger.debug("跳过过小 crop: %dx%d (track=%d)", crop_w, crop_h, track_id)
                continue

            # 对 crop 做尺寸归一化：小的放大、大的缩小，统一到 256~512px

            target_min = 256
            target_max = 512
            max_dim = max(crop_w, crop_h)
            if max_dim < target_min:
                scale = target_min / max_dim
                crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)),
                                  interpolation=cv2.INTER_LANCZOS4)
            elif max_dim > target_max:
                scale = target_max / max_dim
                crop = cv2.resize(crop, (int(crop_w * scale), int(crop_h * scale)),
                                  interpolation=cv2.INTER_AREA)

            detections.append(Detection(
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                crop=crop,
            ))

        if detections:
            logger.debug(
                "帧 %d: 检测到 %d 个目标 (IDs: %s)",
                frame_id,
                len(detections),
                [d.track_id for d in detections],
            )

        return detections

    @property
    def model(self):
        return self._model

    def cleanup(self) -> None:
        """清理临时文件。"""
        if self._tracker_tmp_file:
            try:
                Path(self._tracker_tmp_file).unlink(missing_ok=True)
            except Exception:
                pass
            self._tracker_tmp_file = None

    @staticmethod
    def _patch_ultralytics_cfg() -> None:
        """
        修补 ultralytics 配置类，注入可能缺失的 fuse_score 属性。
        
        某些 ultralytics 版本的代码引用了 fuse_score，但 default.yaml 未更新，
        导致 IterableSimpleNamespace 缺少该属性而崩溃。
        """
        try:
            from ultralytics.cfg import IterableSimpleNamespace
            _orig_init = IterableSimpleNamespace.__init__

            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                if not hasattr(self, "fuse_score"):
                    self.fuse_score = False

            # 只补一次，避免重复 patch
            if not getattr(IterableSimpleNamespace.__init__, "_fuse_score_patched", False):
                IterableSimpleNamespace.__init__ = _patched_init
                IterableSimpleNamespace.__init__._fuse_score_patched = True
                logger.info("已修补 ultralytics IterableSimpleNamespace（注入 fuse_score=False）")
        except Exception as e:
            logger.debug("ultralytics 配置修补跳过: %s", e)

    def __del__(self) -> None:
        self.cleanup()
