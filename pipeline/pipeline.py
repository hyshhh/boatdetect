"""
ShipPipeline — 主流水线编排

级联模式（concurrent_mode=false）：
  YOLO 检测 → VLM识别 → 查库+语义检索 → 绑定结果 → 绘制输出

并发模式（concurrent_mode=true）：
  YOLO 检测 → crop 送入队列 → Agent 独立线程异步推理
  → 结果按帧时间戳严格顺序出队 → 匹配到对应帧绘制输出

双层并发架构：
  外层：帧级任务队列（max_queued_frames 限制深度，防 OOM）
  内层：crop 级 API 并发（max_concurrent 控制）
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from agent import AgentResult
from pipeline.detector import ShipDetector, Detection
from pipeline.demo import DemoRenderer
from pipeline.output import ScreenshotSaver
from pipeline.fps import FPSMeter, LatencyMeter
from pipeline.tracker import TrackManager
from pipeline.video_input import InputSource

logger = logging.getLogger(__name__)

# 抑制第三方库的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ShipPipeline:
    """
    船弦号识别视频处理流水线。

    整合 YOLO 检测、Agent 推理、跟踪管理，支持级联/并发双模式。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: 全局配置字典。None 则从 config.yaml 加载。
        """
        if config is None:
            from config import load_config
            config = load_config()

        self._config = config

        # 读取 pipeline 相关配置
        pipe_cfg = config.get("pipeline", {})
        self._concurrent_mode: bool = bool(pipe_cfg.get("concurrent_mode", False))
        self._max_concurrent: int = pipe_cfg.get("max_concurrent") or 4
        self._max_queued_frames: int = pipe_cfg.get("max_queued_frames") or 30
        self._process_every_n: int = max(1, pipe_cfg.get("process_every_n_frames") or 1)
        self._detect_every_n: int = max(1, pipe_cfg.get("detect_every_n_frames") or 1)
        self._demo_enabled: bool = bool(pipe_cfg.get("demo", False))
        self._save_screenshots: bool = bool(pipe_cfg.get("save_screenshots", True))
        self._use_agent: bool = bool(pipe_cfg.get("use_agent", False))
        self._enable_refresh: bool = bool(pipe_cfg.get("enable_refresh", False))
        self._gap_num: int = pipe_cfg.get("gap_num") or 150
        self._prompt_mode: str = pipe_cfg.get("prompt_mode") or "detailed"

        # 读取 Agent 数据库配置
        from database import ShipDatabase
        self._db = ShipDatabase(config=config)

        # Agent 模式：初始化 LangChain ReAct Agent
        self._agent = None
        if self._use_agent:
            from agent import create_agent
            self._agent = create_agent(config=config)
            logger.info("Agent 模式已启用：使用 LangChain ReAct Agent（lookup + retrieve 两步工具链）")

        # 初始化组件
        self._detector = ShipDetector(
            model_path=pipe_cfg.get("yolo_model", "yolov8n.pt"),
            device=pipe_cfg.get("device", ""),
            conf_threshold=pipe_cfg.get("conf_threshold", 0.25),
            tracker_type=pipe_cfg.get("tracker", "bytetrack"),
            tracker_params=pipe_cfg.get("tracker_params"),
            classes=pipe_cfg.get("detect_classes", [8]),  # COCO: 8=boat
        )

        self._tracker = TrackManager(
            max_stale_frames=pipe_cfg.get("max_stale_frames", 300),
        )

        self._fps = FPSMeter(window_seconds=10.0)
        self._latency = LatencyMeter(window_seconds=10.0)

        # Demo 渲染器
        self._renderer = DemoRenderer(
            show_fps=True,
            show_track_id=True,
        )

        # 截图保存器
        output_dir = pipe_cfg.get("output_dir", "./output")
        self._saver = ScreenshotSaver(output_dir=output_dir)

        # 并发模式相关
        self._task_queue: queue.Queue = queue.Queue(
            maxsize=self._max_queued_frames
        )
        self._result_queue: queue.Queue = queue.Queue(maxsize=self._max_queued_frames)
        self._agent_workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # Agent 运行链路日志（限制最大条数防内存泄漏）
        self._agent_trace: list[dict[str, Any]] = []
        self._trace_lock = threading.Lock()
        self._max_trace_entries = 500

        logger.info(
            "ShipPipeline 初始化: mode=%s, inference=%s, process_every=%d, refresh=%s(gap=%d)",
            "concurrent" if self._concurrent_mode else "cascade",
            "agent" if self._use_agent else "hardcoded",
            self._process_every_n,
            "on" if self._enable_refresh else "off",
            self._gap_num,
        )

    # ── Agent 链路日志 ──────────────────────────

    def _log_agent_trace(
        self,
        event_type: str,
        track_id: int,
        frame_id: int,
        content: str = "",
        **extra: Any,
    ) -> None:
        """记录 Agent 运行链路到内存 trace（不直接打印，由 _log_track_summary 统一汇总）。"""
        entry = {
            "type": event_type,
            "track_id": track_id,
            "frame_id": frame_id,
            "content": content,
            "timestamp": time.time(),
            **extra,
        }
        with self._trace_lock:
            self._agent_trace.append(entry)
            # 超过上限时截断，保留最近一半
            if len(self._agent_trace) > self._max_trace_entries:
                self._agent_trace = self._agent_trace[-(self._max_trace_entries // 2):]

    def _log_track_summary(self, track_id: int) -> None:
        """汇总指定 track 的全部链路步骤，一条日志输出 Step1/Step2/Step3。"""
        with self._trace_lock:
            entries = [e for e in self._agent_trace if e["track_id"] == track_id]

        if not entries:
            return

        # 取最新的一个完整链路（同一个 track 可能被刷新多次）
        # 按 frame_id 降序，取最新一组
        latest_frame = max(e["frame_id"] for e in entries)
        entries = [e for e in entries if e["frame_id"] == latest_frame]

        frame_id = latest_frame
        types = {e["type"]: e["content"] for e in entries}

        # Step1: VLM 预识别
        step1 = types.get("agent_vlm_preinfer") or types.get("cascade_vlm") or "—"
        # Step2: 精确查找
        step2 = types.get("cascade_lookup") or "—"
        # Step3: 语义检索 / 最终结果
        step3 = (types.get("agent_chain_result")
                 or types.get("agent_fallback")
                 or types.get("cascade_result")
                 or types.get("agent_error_fallback")
                 or "—")

        logger.info(
            "[Track %d] frame=%d | Step1(VLM): %s | Step2(Lookup): %s | Step3(Result): %s",
            track_id, frame_id, step1, step2, step3,
        )

    # ── 工具方法 ──────────────────────────────────

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """将 BGR 图像编码为 base64 字符串。"""
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _run_three_step_chain(
        self,
        crop: np.ndarray,
        track_id: int = 0,
        frame_id: int = 0,
    ) -> AgentResult:
        """
        执行三步链路：VLM识别 → 精确查找 → 语义检索。

        第1步: _vlm_infer(crop) → 弦号 + 描述
        第2步: db.lookup(hull_number) → 有弦号时精确查找
        第3步: db.semantic_search_filtered(description) → 弦号未匹配或无弦号时语义检索
        """
        from tools import _vlm_infer

        # 第一步：VLM 识别
        crop_b64 = self._encode_image(crop)
        vlm_result = _vlm_infer(crop_b64, prompt_mode=self._prompt_mode)
        hull_number = vlm_result.get("hull_number", "")
        description = vlm_result.get("description", "")

        self._log_agent_trace(
            "cascade_vlm",
            track_id=track_id,
            frame_id=frame_id,
            content=f"弦号={hull_number or '(无)'} 描述={description[:40] if description else '(无)'}",
        )

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回结果")

        return self._local_lookup_retrieve(hull_number, description, track_id=track_id, frame_id=frame_id)

    def _local_lookup_retrieve(
        self,
        hull_number: str,
        description: str,
        track_id: int = 0,
        frame_id: int = 0,
    ) -> AgentResult:
        """
        本地查库 + 语义检索（不含 VLM 调用）。
        供 _run_three_step_chain 和 _run_agent_chain fallback 使用。
        """
        exact_matched = False
        semantic_ids: list[str] = []

        if hull_number:
            desc_in_db = self._db.lookup(hull_number)
            if desc_in_db is not None:
                exact_matched = True
                description = description or desc_in_db
            elif description:
                results = self._db.semantic_search_filtered(description)
                semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]
        elif description:
            results = self._db.semantic_search_filtered(description)
            semantic_ids = [r["hull_number"] for r in results if r.get("hull_number")]

        match_type = "exact" if exact_matched else ("semantic" if semantic_ids else "none")

        if track_id:
            self._log_agent_trace(
                "cascade_lookup",
                track_id=track_id,
                frame_id=frame_id,
                content=f"精确查找: {'命中' if exact_matched else '未命中'}",
            )
            self._log_agent_trace(
                "cascade_result",
                track_id=track_id,
                frame_id=frame_id,
                content=f"弦号={hull_number or '(无)'} 匹配={match_type} 语义候选={semantic_ids}",
            )

        return AgentResult(
            hull_number=hull_number,
            description=description,
            match_type=match_type,
            semantic_match_ids=semantic_ids,
        )

    def _run_agent_chain(self, crop: np.ndarray, track_id: int = 0, frame_id: int = 0) -> AgentResult:
        """
        Agent 模式执行链路。

        1. 本地 VLM 预识别（一次调用，结果以文本传入 Agent）
        2. Agent 编排 lookup + retrieve 两步工具链（不再调用 VLM）

        Agent 的工具集不含 recognize_ship，避免重复 VLM 调用。
        """
        if self._agent is None:
            raise RuntimeError("Agent 模式未初始化（use_agent=True 但 agent 为 None）")

        from tools import _vlm_infer

        crop_b64 = self._encode_image(crop)
        vlm_result = _vlm_infer(crop_b64, prompt_mode=self._prompt_mode)
        hull_number = vlm_result.get("hull_number", "")
        description = vlm_result.get("description", "")

        self._log_agent_trace(
            "agent_vlm_preinfer",
            track_id=track_id,
            frame_id=frame_id,
            content=f"VLM预识别: 弦号={hull_number or '(无)'} 描述={description[:50] if description else '(无)'}",
        )

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回结果")

        # 将 VLM 结果以文本传入 Agent，Agent 只做 lookup + retrieve（不含 recognize_ship）
        query = (
            f"VLM 识别结果：弦号=\"{hull_number}\"，描述=\"{description}\"。"
            f"请按步骤查找数据库。"
        )

        try:
            result = self._agent.run_with_result(query)
        except Exception as e:
            logger.exception("Agent 执行异常，回退本地检索")
            self._log_agent_trace("agent_error_fallback", track_id, frame_id, content=str(e))
            return self._local_lookup_retrieve(hull_number, description, track_id=track_id, frame_id=frame_id)

        # Agent 结果兜底：若 Agent 未返回 hull_number 但 VLM 有识别值，用 VLM 的值
        if not result.hull_number and hull_number:
            result.hull_number = hull_number

        # 记录 Agent 的 lookup 结果（补 Step2 日志）
        if result.match_type == "exact":
            self._log_agent_trace(
                "cascade_lookup", track_id=track_id, frame_id=frame_id,
                content=f"精确查找: 命中 (弦号={result.hull_number})",
            )
        elif hull_number:
            self._log_agent_trace(
                "cascade_lookup", track_id=track_id, frame_id=frame_id,
                content=f"精确查找: 未命中 (VLM弦号={hull_number} 不在库内)",
            )

        # Agent 返回了有效结果，直接使用
        if result.hull_number or result.semantic_match_ids:
            self._log_agent_trace(
                "agent_chain_result", track_id, frame_id,
                content=f"弦号={result.hull_number or '(无)'} 匹配={result.match_type} 语义候选={result.semantic_match_ids}",
            )
            return result

        # Agent 无结果（可能是 LLM 没按指示调用工具），本地检索兜底
        self._log_agent_trace("agent_fallback", track_id, frame_id, content="Agent 无结果，本地检索兜底")
        return self._local_lookup_retrieve(hull_number, description, track_id=track_id, frame_id=frame_id)

    def _run_recognition(self, crop: np.ndarray, track_id: int = 0, frame_id: int = 0) -> AgentResult:
        """
        统一识别调度：根据 use_agent 配置选择硬编码链路或 Agent 工具链。

        - use_agent=False → _run_three_step_chain（直接调用 VLM HTTP API + 查库 + 语义检索）
        - use_agent=True  → _run_agent_chain（VLM 预识别 + Agent 编排 lookup → retrieve）
        """
        with self._latency.measure("agent"):
            if self._use_agent:
                return self._run_agent_chain(crop, track_id=track_id, frame_id=frame_id)
            return self._run_three_step_chain(crop, track_id=track_id, frame_id=frame_id)

    # ── 推理结果处理 ────────────────────────────

    def _handle_agent_result(
        self,
        track_id: int,
        frame_id: int,
        agent_result: AgentResult,
    ) -> None:
        """处理识别结果：绑定到 track。"""
        # 一条汇总日志：Step1(VLM) / Step2(Lookup) / Step3(Result)
        self._log_track_summary(track_id)

        # 绑定识别结果
        self._tracker.bind_result(
            track_id,
            agent_result.hull_number,
            agent_result.description,
            frame_id=frame_id,
        )

        if agent_result.match_type == "exact":
            # 精确匹配 → 绿色（description 已在链路中获取，直接使用）
            self._tracker.bind_db_match(
                track_id,
                agent_result.hull_number,
                agent_result.description,
            )
        elif agent_result.semantic_match_ids:
            # 有语义匹配候选 → 黄色或红色
            self._tracker.bind_semantic_matches(track_id, agent_result.semantic_match_ids)

    def _handle_agent_error(
        self,
        track_id: int,
        frame_id: int,
        error: str,
    ) -> None:
        """处理 Agent 推理错误：绑定空结果，避免 track 卡在 pending/未识别状态无限重试。"""
        logger.warning("Agent 推理出错 (track=%d, frame=%d): %s", track_id, frame_id, error)
        self._tracker.bind_result(
            track_id,
            hull_number="",
            description="",
            frame_id=frame_id,
        )
        self._log_agent_trace(
            "agent_error_bound_empty",
            track_id=track_id,
            frame_id=frame_id,
            content=f"错误绑定空结果: {error[:80]}",
        )
        self._log_track_summary(track_id)

    # ── 级联模式 ────────────────────────────────

    def _cascade_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """级联模式：同步处理每个需要识别的检测目标。"""
        for det in detections:
            if det.crop is None or det.crop.size == 0:
                continue

            # 判断是否需要识别：新 track 或定时刷新
            need_new = self._tracker.needs_recognition(det.track_id)
            need_refresh = (
                self._enable_refresh
                and self._tracker.needs_refresh(det.track_id, frame_id, self._gap_num)
            )

            if not need_new and not need_refresh:
                continue

            self._tracker.mark_pending(det.track_id)

            trace_type = "cascade_refresh" if need_refresh else "cascade_infer_start"
            self._log_agent_trace(
                trace_type,
                track_id=det.track_id,
                frame_id=frame_id,
                content="定时刷新推理" if need_refresh else "同步推理开始",
            )

            try:
                agent_result = self._run_recognition(det.crop, track_id=det.track_id, frame_id=frame_id)
                self._log_agent_trace(
                    "recognition_result",
                    track_id=det.track_id,
                    frame_id=frame_id,
                    content=(
                        f"弦号={agent_result.hull_number or '(无)'} "
                        f"匹配={agent_result.match_type} "
                        f"语义候选={agent_result.semantic_match_ids}"
                    ),
                )
                self._handle_agent_result(det.track_id, frame_id, agent_result)
            except Exception as e:
                self._handle_agent_error(det.track_id, frame_id, str(e))

    # ── 并发模式 ────────────────────────────────

    def _concurrent_process(
        self,
        detections: list[Detection],
        frame_id: int,
    ) -> None:
        """并发模式：将 crop 送入队列，Agent 异步推理。队列半满时跳过入队（背压）。"""
        # 背压：队列积压超过半满，跳过本轮入队，让 worker 追上
        if self._task_queue.qsize() > self._max_queued_frames // 2:
            logger.debug("队列半满 (%d/%d)，跳过本轮入队", self._task_queue.qsize(), self._max_queued_frames)
            return

        for det in detections:
            if det.crop is None or det.crop.size == 0:
                continue

            # 判断是否需要识别：新 track 或定时刷新
            need_new = self._tracker.needs_recognition(det.track_id)
            need_refresh = (
                self._enable_refresh
                and self._tracker.needs_refresh(det.track_id, frame_id, self._gap_num)
            )

            if not need_new and not need_refresh:
                continue

            # 标记为 pending
            self._tracker.mark_pending(det.track_id)

            task = {
                "frame_id": frame_id,
                "timestamp": time.time(),
                "track_id": det.track_id,
                "crop": det.crop.copy(),
            }

            try:
                self._task_queue.put_nowait(task)
                trace_type = "concurrent_refresh_enqueue" if need_refresh else "concurrent_enqueue"
                self._log_agent_trace(
                    trace_type,
                    track_id=det.track_id,
                    frame_id=frame_id,
                    content=f"{'定时刷新' if need_refresh else '新track'}送入异步队列 (队列深度: {self._task_queue.qsize()})",
                )
            except queue.Full:
                logger.warning(
                    "任务队列已满 (%d)，丢弃 frame=%d track=%d",
                    self._max_queued_frames, frame_id, det.track_id,
                )
                # 取消 pending 状态
                self._tracker.cancel_pending(det.track_id)

    def _agent_worker_loop(self) -> None:
        """Agent 工作线程：从队列取任务并推理。"""
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                track_id = task["track_id"]
                frame_id = task["frame_id"]
                crop = task["crop"]

                self._log_agent_trace(
                    "concurrent_infer_start",
                    track_id=track_id,
                    frame_id=frame_id,
                    content="异步推理开始",
                )

                try:
                    agent_result = self._run_recognition(crop, track_id=track_id, frame_id=frame_id)
                except Exception as e:
                    logger.exception("Agent 推理异常 (track=%d, frame=%d)", track_id, frame_id)
                    agent_result = AgentResult(answer=str(e))

                try:
                    self._result_queue.put_nowait({
                        "frame_id": frame_id,
                        "track_id": track_id,
                        "agent_result": agent_result,
                    })
                except queue.Full:
                    logger.warning("结果队列已满，丢弃结果 (track=%d, frame=%d)", track_id, frame_id)
                    # 结果队列满时直接绑定空结果，避免 track 卡死
                    self._tracker.bind_result(track_id, hull_number="", description="", frame_id=frame_id)
        except Exception:
            logger.exception("Agent 工作线程意外退出")

    def _drain_results(self) -> int:
        """排空结果队列，处理所有已完成的异步推理结果。返回处理数量。"""
        count = 0
        while True:
            try:
                pending = self._result_queue.get_nowait()
                track_id = pending["track_id"]
                frame_id = pending["frame_id"]
                agent_result = pending["agent_result"]

                # 有任何有效信息（弦号/语义匹配/非失败回答）都算有效结果
                if (agent_result.hull_number
                        or agent_result.semantic_match_ids
                        or agent_result.match_type == "exact"
                        or agent_result.match_type == "semantic"):
                    self._handle_agent_result(track_id, frame_id, agent_result)
                else:
                    self._handle_agent_error(track_id, frame_id, agent_result.answer or "Agent 无结果")
                count += 1
            except queue.Empty:
                break
        return count

    def _start_agent_workers(self) -> None:
        """启动 Agent 工作线程池。"""
        self._stop_event.clear()
        self._agent_workers.clear()
        for i in range(self._max_concurrent):
            worker = threading.Thread(
                target=self._agent_worker_loop,
                name=f"agent-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._agent_workers.append(worker)
        logger.info("启动 %d 个 Agent 工作线程", self._max_concurrent)

    def _stop_agent_workers(self) -> None:
        """停止 Agent 工作线程，等待全部完成。"""
        self._stop_event.set()
        for worker in self._agent_workers:
            worker.join(timeout=10.0)
            if worker.is_alive():
                logger.warning("工作线程 %s 未在超时内退出", worker.name)
        self._agent_workers.clear()

        # workers 已停止，排空未处理任务
        while True:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break

        # 排空残留结果（worker 可能在 stop_event 后完成了最后一个任务）
        remaining = self._drain_results()
        if remaining:
            logger.info("处理 %d 个残留结果", remaining)

        logger.info("Agent 工作线程已停止")

    # ── 渲染 ────────────────────────────────────

    def _render_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        frame_id: int,
    ) -> np.ndarray:
        """通过 DemoRenderer 在帧上绘制检测框、识别结果和 HUD。"""
        return self._renderer.render(
            frame=frame,
            detections=detections,
            tracks=self._tracker.active_tracks,
            fps_info=self._fps.get_all_fps(),
            frame_id=frame_id,
            queue_depth=self._task_queue.qsize(),
            max_queue=self._max_queued_frames,
        )

    # ── 主流程 ──────────────────────────────────

    def process(
        self,
        source: str | int,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        运行完整的视频处理流水线。

        Args:
            source: 视频输入源（文件路径/相机号/RTSP URL）。
            output_path: 输出视频路径（可选）。
            display: 是否实时显示窗口（仅本地有显示器时有效）。
            max_frames: 最大处理帧数，0 表示不限制。
            frame_callback: 每帧处理完成后的回调函数 callback(frame, frame_id)。

        Returns:
            统计信息字典。
        """
        input_src = InputSource(source)
        video_writer = None
        last_detections: list[Detection] = []
        frame_id = 0
        total_detections = 0
        start_time = time.time()

        try:
            # 初始化视频写入器
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    output_path, fourcc,
                    input_src.source_fps,
                    (input_src.width, input_src.height),
                )
                if not video_writer.isOpened():
                    logger.error("无法创建输出视频: %s", output_path)
                    video_writer = None
                else:
                    logger.info("输出视频: %s", output_path)

            # 启动并发 worker
            if self._concurrent_mode:
                self._start_agent_workers()

            logger.info(
                "开始处理: source=%s, mode=%s, inference=%s, demo=%s, refresh=%s(gap=%d), detect_every=%d, process_every=%d",
                source,
                "concurrent" if self._concurrent_mode else "cascade",
                "agent" if self._use_agent else "hardcoded",
                self._demo_enabled,
                "on" if self._enable_refresh else "off",
                self._gap_num,
                self._detect_every_n,
                self._process_every_n,
            )

            while True:
                ret, frame = input_src.read()
                if not ret:
                    logger.info("视频源结束或读取失败")
                    break

                frame_id += 1
                if max_frames > 0 and frame_id > max_frames:
                    break

                # FPS 统计
                self._fps.tick("stream")

                # ── 每 N 帧进行 YOLO 检测，其余帧复用上次结果 ──
                should_detect = (frame_id % self._detect_every_n == 0)

                if should_detect:
                    try:
                        with self._latency.measure("yolo"):
                            detections = self._detector.detect(frame, frame_id)
                    except Exception as e:
                        logger.error("YOLO 检测异常 (frame=%d): %s", frame_id, e)
                        detections = []
                    last_detections = detections
                else:
                    detections = last_detections

                total_detections += len(detections)

                # 注册/更新 track（每帧执行，保持跟踪状态）
                for det in detections:
                    self._tracker.get_or_create(det.track_id, frame_id)

                # ── process_every_n_frames 控制 Agent 推理频率 ──
                should_process = (frame_id % self._process_every_n == 0)

                if should_process:
                    # Agent 推理（级联或并发）
                    if self._concurrent_mode:
                        self._concurrent_process(detections, frame_id)
                    else:
                        self._cascade_process(detections, frame_id)

                # 并发模式下排空已完成的结果（非阻塞）
                if self._concurrent_mode:
                    self._drain_results()

                # 每 30 帧清理一次过期 track（不必每帧扫描）
                if frame_id % 30 == 0:
                    self._tracker.cleanup_stale(frame_id)

                # 渲染输出
                if self._demo_enabled or output_path or display:
                    with self._latency.measure("demo"):
                        display_frame = self._render_frame(frame, last_detections, frame_id)
                else:
                    display_frame = frame

                # 每 N 帧：有已识别的 track 就保存截图（需开启 save_screenshots）
                if self._save_screenshots and should_process:
                    active = self._tracker.active_tracks
                    if any(t.recognized for t in active.values()):
                        self._saver.save(display_frame, frame_id)

                # 写入输出视频
                if video_writer:
                    video_writer.write(display_frame)

                # 回调
                if frame_callback:
                    frame_callback(display_frame, frame_id)

                # 实时显示
                if display:
                    cv2.imshow("Ship Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("用户按下 q，停止处理")
                        break

                # 处理 FPS
                self._fps.tick("process")

                # 定期打印状态（每 5 秒，一条汇总）
                if self._fps.should_print("stream"):
                    elapsed = time.time() - start_time
                    stream_fps = self._fps.get_fps("stream")
                    process_fps = self._fps.get_fps("process")

                    # Latency 部分
                    latency_parts = []
                    for stage in ("yolo", "agent", "demo"):
                        s = self._latency.get_stats(stage)
                        if s and s["count"] > 0:
                            latency_parts.append(
                                f"{stage}: avg={s['avg']:.1f}ms p95={s['p95']:.1f}ms (n={s['count']})"
                            )
                    latency_str = f" | Latency: {' | '.join(latency_parts)}" if latency_parts else ""

                    # 近期 track 部分（每 100 帧附带）
                    trace_str = ""
                    if frame_id % 100 < self._process_every_n:
                        with self._trace_lock:
                            recent_tracks = len(set(e["track_id"] for e in self._agent_trace[-50:])) if self._agent_trace else 0
                        if recent_tracks:
                            trace_str = f" | 近期处理: {recent_tracks} tracks"

                    logger.info(
                        "FPS: stream=%.1f process=%.1f | frames=%d elapsed=%ds tracks=%d%s%s",
                        stream_fps, process_fps, frame_id, int(elapsed),
                        len(self._tracker), latency_str, trace_str,
                    )

            # ── 处理完成，收集统计 ──

            # 并发模式下最终排空结果
            if self._concurrent_mode:
                self._drain_results()

            elapsed = time.time() - start_time
            tracks = self._tracker.active_tracks
            total_recognized = sum(1 for t in tracks.values() if t.recognized)

            stats = {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(tracks),
                "recognized_tracks": total_recognized,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "inference": "agent" if self._use_agent else "hardcoded",
                "screenshots_saved": self._saver.saved_count,
                "latency": self._latency.get_all_stats(),
            }

            logger.info("=" * 50)
            logger.info("处理完成统计:")
            logger.info("  总帧数: %d", stats["total_frames"])
            logger.info("  总检测数: %d", stats["total_detections"])
            logger.info("  跟踪目标数: %d", stats["total_tracks"])
            logger.info("  已识别: %d", stats["recognized_tracks"])
            logger.info("  耗时: %.1fs", stats["elapsed_seconds"])
            logger.info("  平均 FPS: %.1f", stats["avg_fps"])
            logger.info("  模式: %s", stats["mode"])
            logger.info("=" * 50)

            return stats

        except KeyboardInterrupt:
            logger.info("用户中断处理")
            elapsed = time.time() - start_time
            return {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(self._tracker),
                "recognized_tracks": 0,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "interrupted": True,
            }

        finally:
            if self._concurrent_mode:
                self._stop_agent_workers()
            input_src.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            self._detector.cleanup()

    # ── 链路摘要 ────────────────────────────────

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        """获取完整的 Agent 运行链路日志。"""
        with self._trace_lock:
            return list(self._agent_trace)

    # ── 运行时控制 ──────────────────────────────

    def set_demo(self, enabled: bool) -> None:
        """设置 demo 开关。"""
        self._demo_enabled = enabled
        logger.info("Demo 模式: %s", "开启" if enabled else "关闭")

    def set_prompt_mode(self, mode: str) -> None:
        """设置提示词模式：detailed（详细）或 brief（简略）。"""
        if mode not in ("detailed", "brief"):
            raise ValueError(f"不支持的提示词模式: {mode}，仅支持 detailed/brief")
        self._prompt_mode = mode
        logger.info("提示词模式切换为: %s", mode)

    def set_use_agent(self, enabled: bool) -> None:
        """设置 Agent 模式开关。"""
        if enabled and self._agent is None:
            from agent import create_agent
            self._agent = create_agent(config=self._config)
            logger.info("Agent 模式已启用：初始化 LangChain ReAct Agent")
        self._use_agent = enabled
        logger.info("推理模式: %s", "Agent (LangChain)" if enabled else "硬编码 (直接调用)")

    def switch_to_concurrent(self, enabled: bool) -> None:
        """动态切换级联/并发模式。"""
        self._concurrent_mode = enabled
        logger.info("切换为 %s 模式", "并发" if enabled else "级联")
