"""Agent 核心 — 构建与运行，两步链路：精确查找 → 语义检索（VLM 由外层预调用）"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from config import load_config
from database import ShipDatabase
from tools import build_tools

logger = logging.getLogger(__name__)

# ── System Prompt ──────────────────────────────

SYSTEM_PROMPT = """你是船弦号识别助手。输入中已包含 VLM 预识别的弦号和描述，直接执行以下两步：

【两步链路】
1. 如果输入中有弦号（非空），调用 lookup_by_hull_number 精确查找
   → found=true：直接返回「库内确定id：{hull_number}，描述：{description}」
   → found=false：进入第二步

2. 调用 retrieve_by_description 语义检索（用输入中的 description 作为查询）
   → 有结果：返回「可能id：{弦号1}/{弦号2}/...」
   → 无结果：返回「未识别」

如果输入中弦号为空，直接跳过第一步进入第二步。

【禁止】
- 不要编造弦号或描述
- 不要跳过任何步骤
- 不要同时调用多个工具
"""

# ── 无 Few-shot 示例（避免误导 Agent）──


class AgentResult:
    """Agent 运行结果，包含结构化信息供 pipeline 使用。"""

    def __init__(
        self,
        hull_number: str = "",
        description: str = "",
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        answer: str = "",
    ):
        self.hull_number = hull_number
        self.description = description
        self.match_type = match_type        # "exact" | "semantic" | "none"
        self.semantic_match_ids = semantic_match_ids or []
        self.answer = answer


class ShipHullAgent:
    """船弦号识别 Agent 封装。两步链路：lookup → retrieve（VLM 由外层预调用）。"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_config()

        app_cfg = self.config.get("app", {})
        llm_cfg = self.config.get("llm", {})

        logging.basicConfig(
            level=getattr(logging, app_cfg.get("log_level", "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        self.db = ShipDatabase(config=self.config)
        # Pipeline 模式：不含 recognize_ship（VLM 由外层预调用）
        self.tools = build_tools(self.db, include_recognize=False)

        self._llm = ChatOpenAI(
            model=llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ"),
            api_key=llm_cfg.get("api_key", "abc123"),
            base_url=llm_cfg.get("base_url", "http://localhost:7890/v1"),
            temperature=llm_cfg.get("temperature", 0.0),
        )

        self._agent = create_react_agent(
            model=self._llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

    def run(self, query: str) -> str:
        """运行 Agent，返回自然语言回答。"""
        logger.info("收到查询: %s", query[:100])
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            logger.info("回答: %s", answer[:100])
            return answer
        except Exception as e:
            logger.exception("Agent 执行失败")
            return f"查询执行失败: {e}"

    def run_with_result(self, query: str) -> AgentResult:
        """运行 Agent，返回结构化结果（供 pipeline 使用）。"""
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            return self._parse_result(result)
        except Exception as e:
            logger.exception("Agent 执行失败")
            return AgentResult(answer=f"Agent 执行失败: {e}")

    @staticmethod
    def _parse_result(result: dict) -> AgentResult:
        """从 Agent 消息历史中提取结构化结果。"""
        msgs = result.get("messages", [])
        hull_number = ""
        description = ""
        match_type = "none"
        semantic_match_ids: list[str] = []
        answer = msgs[-1].content if msgs else ""

        for msg in msgs:
            if not isinstance(msg, ToolMessage):
                continue
            try:
                data = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue

            # recognize_ship 结果
            if "hull_number" in data and "description" in data and "found" not in data and "results" not in data:
                hull_number = data.get("hull_number", "")
                description = data.get("description", "")

            # lookup_by_hull_number（无论 found 与否都保留 hull_number）
            if "found" in data:
                if data.get("found") is True:
                    match_type = "exact"
                    description = data.get("description", description)
                # found=false 时也要保留 hull_number（来自 VLM 识别）
                hull_number = data.get("hull_number", hull_number) or hull_number

            # retrieve_by_description 语义匹配
            if "results" in data:
                results = data["results"]
                if results:
                    semantic_match_ids = [
                        r.get("hull_number", "") for r in results if r.get("hull_number")
                    ]
                    if match_type != "exact":
                        match_type = "semantic"

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_match_ids,
            answer=answer,
        )

    def run_verbose(self, query: str) -> list[dict]:
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=query)]})
            trace = []
            for msg in result["messages"]:
                entry = {"type": msg.type, "content": msg.content}
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    entry["tool_calls"] = [
                        {"name": tc["name"], "args": tc["args"]}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "tool_call_id"):
                    entry["tool_call_id"] = msg.tool_call_id
                trace.append(entry)
            return trace
        except Exception as e:
            logger.exception("Agent 执行失败")
            return [{"type": "error", "content": f"查询执行失败: {e}"}]


_agent_instance: ShipHullAgent | None = None
_agent_config_hash: int = 0
_agent_lock = threading.Lock()


def create_agent(config: dict[str, Any] | None = None) -> ShipHullAgent:
    global _agent_instance, _agent_config_hash
    config_hash = hash(str(config)) if config is not None else 0
    with _agent_lock:
        if _agent_instance is None or config_hash != _agent_config_hash:
            _agent_instance = ShipHullAgent(config)
            _agent_config_hash = config_hash
    return _agent_instance
