"""LangChain 工具定义 — 可选 recognize_ship + lookup_by_hull_number + retrieve_by_description"""

from __future__ import annotations

import json
import logging
import re
from typing import Annotated

import cv2
import httpx
import numpy as np
from langchain_core.tools import tool

from database import ShipDatabase
from config import load_config

logger = logging.getLogger(__name__)

# ── 配置缓存（避免每次推理都读磁盘）──
_cached_llm_cfg: dict | None = None


def _get_llm_cfg() -> dict:
    """获取 LLM 配置（带缓存）。"""
    global _cached_llm_cfg
    if _cached_llm_cfg is None:
        config = load_config()
        _cached_llm_cfg = config.get("llm", {})
    return _cached_llm_cfg


def _vlm_infer(image_b64: str, prompt_mode: str = "detailed") -> dict:
    """调用 VLM 进行弦号识别，返回 {hull_number, description}。"""
    llm_cfg = _get_llm_cfg()

    api_url = f"{llm_cfg.get('base_url', 'http://localhost:7890/v1').rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {llm_cfg.get('api_key', 'abc123')}",
        "Content-Type": "application/json",
    }

    # 解码并重新编码 base64 图像，提高 JPEG 质量（识别弦号文字需要更高清晰度）
    import base64 as _b64
    try:
        img_bytes = _b64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_b64 = _b64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        pass  # 解码失败则使用原始 base64

    if prompt_mode == "brief":
        prompt = (
            "识别船体上的弦号编号，不要评价图片质量。\n"
            "返回 JSON（不要其他文字）：\n"
            '{"hull_number": "弦号编号（无则空字符串）", '
            '"description": "简要描述：船型+颜色+主要特征（50字内，不提图片质量）"}'
        )
    else:
        prompt = (
            "你是船只弦号识别专家。你的核心任务是读取船体侧面的文字编号。\n\n"
            "重要指令：\n"
            "- 不要评价图片质量（无论清晰还是模糊都不要提）\n"
            "- 不要说\"看不清\"\"质量低\"等废话\n"
            "- 即使图片模糊，也必须尝试读取船体上的任何可见文字、数字、编号\n"
            "- 重点关注：船体侧面白色/黑色的编号区域、船尾文字、船名\n\n"
            "返回 JSON（不要其他文字）：\n"
            '{"hull_number": "读到的弦号编号（如 0014、海巡123、A01 等，完全没有可见文字则返回空字符串）", '
            '"description": "客观描述船只：船型+船体颜色+上层建筑颜色+特殊标志（不提图片质量）"}'
        )

    payload = {
        "model": llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
        "temperature": llm_cfg.get("temperature", 0.0),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }

    resp = httpx.post(api_url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        logger.error("VLM 返回格式异常: %s, 原始: %s", e, resp.text[:300])
        return {"hull_number": "", "description": ""}

    # 解析 JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    result: dict = {}
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("VLM 返回无法解析为 JSON: %s", content[:200])
        else:
            logger.warning("VLM 返回无 JSON 结构: %s", content[:200])

    if not isinstance(result, dict):
        logger.warning("VLM 返回非字典类型: %s", type(result).__name__)
        result = {}

    return {
        "hull_number": str(result.get("hull_number") or "").strip(),
        "description": str(result.get("description") or "").strip(),
    }


def build_tools(db: ShipDatabase, include_recognize: bool = True) -> list:
    """构建链路工具。

    Args:
        db: 船只数据库实例。
        include_recognize: 是否包含 recognize_ship 工具。
            True  = 独立运行模式（三步链路：recognize → lookup → retrieve）
            False = Pipeline 模式（两步链路：lookup → retrieve，VLM 由外层预调用）
    """
    tools_list: list = []

    if include_recognize:
        @tool
        def recognize_ship(
            image_base64: Annotated[str, "裁剪的船只图像 base64 编码字符串（JPEG）"],
        ) -> str:
            """
            第一步：对船只图像进行弦号识别。
            调用视觉大模型分析图像，返回识别到的弦号和船只描述。
            有弦号时接下来调 lookup_by_hull_number；无弦号时调 retrieve_by_description。
            """
            try:
                result = _vlm_infer(image_base64)
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                logger.exception("船只识别失败")
                return json.dumps({"error": str(e), "hull_number": "", "description": ""}, ensure_ascii=False)

        tools_list.append(recognize_ship)

    @tool
    def lookup_by_hull_number(
        hull_number: Annotated[str, "要查询的船弦号，例如 '0014'"],
    ) -> str:
        """
        第二步：通过弦号精确查找船只描述。
        recognize_ship 返回有弦号时调用此工具。found=true 则结束；found=false 进入第三步。
        """
        hull_number = hull_number.strip()
        desc = db.lookup(hull_number)
        if desc is not None:
            return json.dumps(
                {"found": True, "hull_number": hull_number, "description": desc},
                ensure_ascii=False,
            )
        return json.dumps({"found": False, "hull_number": hull_number}, ensure_ascii=False)

    @tool
    def retrieve_by_description(
        target_description: Annotated[str, "对目标船的外观文字描述，越详细越好"],
    ) -> str:
        """
        第三步：基于 FAISS 向量库的语义检索。
        当弦号查不到（第二步 found=false），或 recognize_ship 未识别到弦号时调用。
        输入对船只的外观描述，返回最匹配的结果。
        """
        try:
            results = db.semantic_search_filtered(target_description)
            if not results:
                raw = db.semantic_search(target_description)
                if raw:
                    return json.dumps(
                        {"note": "以下结果相似度较低，仅供参考", "results": raw},
                        ensure_ascii=False,
                    )
                return json.dumps({"error": "未找到匹配结果"}, ensure_ascii=False)

            return json.dumps({"results": results}, ensure_ascii=False)
        except Exception as e:
            logger.exception("语义检索失败")
            return json.dumps({"error": f"语义检索失败: {e}"}, ensure_ascii=False)

    tools_list.extend([lookup_by_hull_number, retrieve_by_description])
    return tools_list
