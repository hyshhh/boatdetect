"""
AgentInference — 使用 Qwen3.5 VLM 对裁剪的船只图像进行弦号识别

将 crop 图像编码为 base64，调用 OpenAI 兼容的视觉模型 API，
输出弦号 (hull_number) 和描述 (description)。
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── 提示词模板 ──────────────────────────────────

DETAILED_PROMPT = """你是船只弦号识别专家。你的核心任务是读取船体侧面的文字编号。

重要指令：
- 不要评价图片质量（无论清晰还是模糊都不要提）
- 不要说"看不清""质量低"等废话
- 即使图片模糊，也必须尝试读取船体上的任何可见文字、数字、编号
- 重点关注：船体侧面白色/黑色的编号区域、船尾文字、船名

请返回以下 JSON（不要任何其他文字）：
{{
  "hull_number": "读到的弦号编号（如 0014、海巡123、A01 等，完全没有可见文字则返回空字符串）",
  "description": "客观描述船只：船型+船体颜色+上层建筑颜色+特殊标志（不提图片质量）"
}}"""

BRIEF_PROMPT = """识别船体上的弦号编号，不要评价图片质量。

返回 JSON（不要其他文字）：
{{
  "hull_number": "弦号编号（无则空字符串）",
  "description": "简要描述：船型+颜色+主要特征（50字内，不提图片质量）"
}}"""


@dataclass
class InferenceResult:
    """单次推理结果。"""
    hull_number: str
    description: str
    track_id: int
    frame_id: int
    error: str | None = None


class AgentInference:
    """
    基于视觉大模型的船只弦号识别推理器。

    支持：
    - 详细/简略提示词切换
    - 单张图像同步推理
    - 批量并发推理（线程池）
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        prompt_mode: str = "detailed",
        max_concurrent: int = 4,
    ):
        """
        Args:
            config: 全局配置字典，从中读取 llm 配置。
            prompt_mode: "detailed" 或 "brief"。
            max_concurrent: 最大并发推理数。
        """
        if config is None:
            from config import load_config
            config = load_config()

        self._config = config
        self._prompt_mode = prompt_mode
        self._max_concurrent = max_concurrent

        # 信号量控制并发
        self._semaphore = threading.Semaphore(max_concurrent)

        llm_cfg = config.get("llm", {})
        self._model = llm_cfg.get("model", "Qwen/Qwen3.5-VL")
        self._api_key = llm_cfg.get("api_key", "abc123")
        self._base_url = llm_cfg.get("base_url", "http://localhost:7890/v1")
        self._temperature = llm_cfg.get("temperature", 0.0)

        self._api_url = f"{self._base_url.rstrip('/')}/chat/completions"

        logger.info(
            "AgentInference 初始化: model=%s, prompt_mode=%s, max_concurrent=%d",
            self._model, prompt_mode, max_concurrent,
        )

    def set_prompt_mode(self, mode: str) -> None:
        """切换提示词模式。"""
        if mode not in ("detailed", "brief"):
            raise ValueError(f"prompt_mode 必须是 'detailed' 或 'brief'，收到 '{mode}'")
        self._prompt_mode = mode
        logger.info("提示词模式切换为: %s", mode)

    @property
    def prompt_mode(self) -> str:
        return self._prompt_mode

    def _get_prompt(self) -> str:
        """获取当前提示词。"""
        if self._prompt_mode == "brief":
            return BRIEF_PROMPT
        return DETAILED_PROMPT

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """将 BGR 图像编码为 base64 字符串。"""
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_response(content: str) -> dict[str, str]:
        """解析模型返回的 JSON。"""
        content = content.strip()

        # 兼容 ```json ... ``` 包裹
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.warning("模型返回非 JSON: %s", content[:200])
                    return {"hull_number": "", "description": content[:200]}
            else:
                logger.warning("模型返回无法解析: %s", content[:200])
                return {"hull_number": "", "description": content[:200]}

        return {
            "hull_number": str(result.get("hull_number") or "").strip(),
            "description": str(result.get("description") or "").strip(),
        }

    def infer_single(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int,
    ) -> InferenceResult:
        """
        对单张裁剪图像进行推理。

        Args:
            crop: 裁剪的船只图像 (BGR)。
            track_id: 跟踪 ID。
            frame_id: 帧编号。

        Returns:
            InferenceResult 包含 hull_number, description。
        """
        with self._semaphore:
            max_retries = 3
            last_err: Exception | None = None

            for attempt in range(max_retries):
                try:
                    return self._infer_single_inner(crop, track_id, frame_id)
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    last_err = e
                    wait = 2 ** attempt
                    logger.warning(
                        "推理网络错误 (track=%d, frame=%d): %s，%ds 后重试 (%d/%d)",
                        track_id, frame_id, e, wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)

            logger.error("推理失败，已重试 %d 次 (track=%d, frame=%d)", max_retries, track_id, frame_id)
            return InferenceResult(
                hull_number="", description="",
                track_id=track_id, frame_id=frame_id,
                error=str(last_err),
            )

    def _infer_single_inner(
        self,
        crop: np.ndarray,
        track_id: int,
        frame_id: int,
    ) -> InferenceResult:
        """单次推理内部实现（不含重试逻辑）。"""
        b64 = self._encode_image(crop)
        prompt = self._get_prompt()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            },
                        },
                    ],
                }
            ],
        }

        resp = httpx.post(
            self._api_url,
            headers=headers,
            json=payload,
            timeout=15,
        )

        if not resp.is_success:
            err_msg = f"API 返回 {resp.status_code}: {resp.text[:300]}"
            logger.error("推理失败 (track=%d, frame=%d): %s", track_id, frame_id, err_msg)
            return InferenceResult(
                hull_number="",
                description="",
                track_id=track_id,
                frame_id=frame_id,
                error=err_msg,
            )

        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            err_msg = f"API 响应解析失败: {e}, 原始响应: {resp.text[:300]}"
            logger.error("推理失败 (track=%d, frame=%d): %s", track_id, frame_id, err_msg)
            return InferenceResult(
                hull_number="",
                description="",
                track_id=track_id,
                frame_id=frame_id,
                error=err_msg,
            )

        parsed = self._parse_response(content)

        logger.info(
            "推理完成 (track=%d, frame=%d): 弦号=%s, 描述=%s",
            track_id, frame_id,
            parsed["hull_number"] or "(未识别)",
            parsed["description"][:50],
        )

        return InferenceResult(
            hull_number=parsed["hull_number"],
            description=parsed["description"],
            track_id=track_id,
            frame_id=frame_id,
        )

    def infer_batch_async(
        self,
        tasks: list[dict[str, Any]],
        callback=None,
    ) -> list[threading.Thread]:
        """
        异步批量推理（启动线程）。

        Args:
            tasks: [{"crop": np.ndarray, "track_id": int, "frame_id": int}, ...]
            callback: 每个任务完成后的回调函数 callback(InferenceResult)。

        Returns:
            启动的线程列表。
        """
        threads = []
        for task in tasks:
            # 默认参数 t=task 捕获当前循环值，避免闭包变量覆盖
            def _worker(t=task, cb=callback):
                result = self.infer_single(
                    crop=t["crop"],
                    track_id=t["track_id"],
                    frame_id=t["frame_id"],
                )
                if cb:
                    cb(result)

            t = threading.Thread(target=_worker, daemon=True)
            t.start()
            threads.append(t)

        return threads
