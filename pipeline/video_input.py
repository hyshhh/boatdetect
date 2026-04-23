"""
InputSource — 视频/相机/视频流统一输入接口

支持：
  - 视频文件（MP4, AVI, MKV 等）
  - USB 相机（设备号）
  - RTSP/HTTP 视频流
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InputSource:
    """
    统一视频输入源。

    根据输入自动判断类型：
    - 数字字符串（如 "0"）→ USB 相机
    - rtsp:// 或 http(s):// 开头 → 网络视频流
    - 文件路径 → 视频文件
    """

    def __init__(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
        buffer_size: int = 1,
    ):
        """
        Args:
            source: 输入源。可以是：
                    - 视频文件路径
                    - 相机设备号（"0", "1"）
                    - RTSP/HTTP URL
            width: 强制设置帧宽度（可选）。
            height: 强制设置帧高度（可选）。
            buffer_size: 视频流缓冲区大小（降低延迟）。
        """
        self._source = source
        self._cap: cv2.VideoCapture | None = None
        self._width = width
        self._height = height
        self._buffer_size = buffer_size
        self._frame_count = 0
        self._is_file = False
        self._total_frames = 0
        self._fps = 0.0

        self._open()

    def _open(self) -> None:
        """打开视频源。"""
        source = self._source

        # 判断输入类型
        if isinstance(source, int):
            cap_source = source
            self._is_file = False
        elif isinstance(source, str) and source.isdigit():
            cap_source = int(source)
            self._is_file = False
        elif isinstance(source, str) and (
            source.startswith("rtsp://") or
            source.startswith("http://") or
            source.startswith("https://")
        ):
            cap_source = source
            self._is_file = False
        else:
            # 视频文件
            p = Path(str(source))
            if not p.exists():
                raise FileNotFoundError(f"视频文件不存在: {p}")
            cap_source = str(p.resolve())
            self._is_file = True

        logger.info("打开视频源: %s (类型: %s)", source, "文件" if self._is_file else "流/相机")

        self._cap = cv2.VideoCapture(cap_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")

        # 设置参数
        if self._width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # 降低缓冲区减少延迟（仅对非文件源有效）
        if not self._is_file:
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
            except Exception:
                pass  # 部分后端不支持设置缓冲区大小

        # 读取元信息
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self._fps <= 0:
            self._fps = 30.0

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "视频源已打开: %dx%d, %.1f FPS, %s帧",
            w, h, self._fps,
            self._total_frames if self._is_file else "未知",
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        读取一帧。

        Returns:
            (success, frame) - success 为 False 表示视频结束或读取失败。
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        if ret:
            self._frame_count += 1
            return True, frame
        return False, None

    def release(self) -> None:
        """释放视频源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("视频源已释放，共处理 %d 帧", self._frame_count)

    @property
    def is_file(self) -> bool:
        return self._is_file

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def source_fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return 0

    @property
    def height(self) -> int:
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return 0
