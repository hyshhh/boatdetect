"""
Output — 截图保存模块

当 process_every_n_frames 触发时，将带有检测框和 Agent 识别结果的
渲染帧保存到 output 目录，用于结果回溯和调试。
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScreenshotSaver:
    """
    截图保存器。

    在指定帧触发时，将渲染后的帧（含检测框、识别结果、HUD）保存为图片。

    用法：
        saver = ScreenshotSaver(output_dir="./output")
        saver.save_if_triggered(rendered_frame, frame_id, process_every_n=30)
    """

    def __init__(
        self,
        output_dir: str | Path = "./output",
        image_format: str = "jpg",
        jpeg_quality: int = 90,
    ):
        """
        Args:
            output_dir: 截图保存目录，自动创建。
            image_format: 图片格式，"jpg" 或 "png"。
            jpeg_quality: JPEG 质量 (1-100)，仅 jpg 格式有效。
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._format = image_format.lower()
        self._jpeg_quality = jpeg_quality
        self._saved_count = 0

        if self._format == "jpg":
            self._ext = ".jpg"
            self._encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        elif self._format == "png":
            self._ext = ".png"
            self._encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            raise ValueError(f"不支持的图片格式: {image_format}，仅支持 jpg/png")

        logger.info(
            "ScreenshotSaver 初始化: dir=%s, format=%s, quality=%d",
            self._output_dir, self._format, jpeg_quality,
        )

    def save(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> str | None:
        """
        直接保存截图（无触发条件判断）。

        Args:
            frame: 已渲染的 BGR 帧。
            frame_id: 当前帧号，用于生成文件名。

        Returns:
            保存的文件路径，失败返回 None。
        """
        filename = f"frame_{frame_id:06d}{self._ext}"
        filepath = self._output_dir / filename

        success = cv2.imwrite(str(filepath), frame, self._encode_params)
        if success:
            self._saved_count += 1
        else:
            logger.error("截图保存失败: %s", filepath)
            return None

        return str(filepath)

    def save_if_triggered(
        self,
        frame: np.ndarray,
        frame_id: int,
        process_every_n: int,
    ) -> str | None:
        """
        检查是否满足触发条件（frame_id % process_every_n == 0），满足则保存截图。

        Args:
            frame: 已渲染的 BGR 帧（含检测框和识别结果）。
            frame_id: 当前帧号。
            process_every_n: 每 N 帧触发一次。

        Returns:
            保存的文件路径，未触发时返回 None。
        """
        if process_every_n <= 0 or frame_id % process_every_n != 0:
            return None

        return self.save(frame, frame_id)

    @property
    def saved_count(self) -> int:
        """已保存的截图总数。"""
        return self._saved_count
