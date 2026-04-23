"""Pipeline 模块单元测试（不需要 GPU / YOLO / LLM API）"""

import time
import pytest
import numpy as np

from pipeline.fps import FPSMeter
from pipeline.tracker import TrackManager
from pipeline.agent_inference import AgentInference


# ══════════════════════════════════════════════
#  FPSMeter 测试
# ══════════════════════════════════════════════

class TestFPSMeter:
    def test_initial_fps_is_zero(self):
        meter = FPSMeter(window_seconds=10.0)
        assert meter.get_fps() == 0.0

    def test_single_tick_zero_fps(self):
        """单次 tick 无法计算 FPS（至少需要 2 次）"""
        meter = FPSMeter(window_seconds=10.0)
        meter.tick("test")
        assert meter.get_fps("test") == 0.0

    def test_multiple_ticks(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(10):
            meter.tick("test")
            time.sleep(0.01)
        fps = meter.get_fps("test")
        assert fps > 0

    def test_multiple_channels(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(5):
            meter.tick("stream")
            meter.tick("process")
        assert "stream" in meter.get_all_fps()
        assert "process" in meter.get_all_fps()

    def test_reset(self):
        meter = FPSMeter(window_seconds=10.0)
        for _ in range(5):
            meter.tick("test")
        meter.reset("test")
        assert meter.get_fps("test") == 0.0

    def test_reset_all(self):
        meter = FPSMeter(window_seconds=10.0)
        meter.tick("a")
        meter.tick("b")
        meter.reset()
        assert meter.get_all_fps() == {}

    def test_should_print(self):
        meter = FPSMeter(window_seconds=10.0)
        meter._print_interval = 0.05  # 缩短间隔以测试
        # 第一次调用 should_print 在 tick 之前会初始化 last_print
        assert meter.should_print("test") is False
        # 经过足够时间后应返回 True
        time.sleep(0.06)
        assert meter.should_print("test") is True
        # 立即再次调用应返回 False（刚打印过）
        assert meter.should_print("test") is False


# ══════════════════════════════════════════════
#  TrackManager 测试
# ══════════════════════════════════════════════

class TestTrackManager:
    def test_get_or_create_new(self):
        mgr = TrackManager()
        info = mgr.get_or_create(1, 100)
        assert info.track_id == 1
        assert info.first_seen_frame == 100
        assert info.recognized is False

    def test_get_or_create_existing(self):
        mgr = TrackManager()
        info1 = mgr.get_or_create(1, 100)
        info2 = mgr.get_or_create(1, 200)
        assert info1 is info2
        assert info2.last_seen_frame == 200

    def test_needs_recognition_new(self):
        mgr = TrackManager()
        assert mgr.needs_recognition(1) is True

    def test_needs_recognition_after_bind(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        assert mgr.needs_recognition(1) is False

    def test_needs_recognition_while_pending(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        assert mgr.needs_recognition(1) is False

    def test_cancel_pending(self):
        """取消 pending 后应恢复为可识别状态。"""
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        assert mgr.needs_recognition(1) is False
        mgr.cancel_pending(1)
        assert mgr.needs_recognition(1) is True

    def test_cancel_pending_nonexistent(self):
        """取消不存在的 track 不应报错。"""
        mgr = TrackManager()
        mgr.cancel_pending(999)  # 不应抛异常

    def test_bind_result(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        info = mgr.active_tracks[1]
        assert info.recognized is True
        assert info.hull_number == "0014"
        assert info.description == "白色大型客轮"
        assert info.pending is False

    def test_bind_db_match(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        mgr.bind_db_match(1, "0014", "白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪")
        info = mgr.active_tracks[1]
        assert info.db_matched is True
        assert info.db_match_id == "0014"

    def test_display_text_waiting(self):
        mgr = TrackManager()
        text = mgr.get_display_text(1)
        assert "等待" in text

    def test_display_text_pending(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.mark_pending(1)
        text = mgr.get_display_text(1)
        assert "识别中" in text

    def test_display_text_db_matched(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "0014", "白色大型客轮")
        mgr.bind_db_match(1, "0014", "白色大型客轮，上层建筑为蓝色涂装")
        text = mgr.get_display_text(1)
        assert "库内确定id" in text
        assert "0014" in text

    def test_display_text_unknown(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.bind_result(1, "X999", "未知小船")
        text = mgr.get_display_text(1)
        assert "未知id" in text

    def test_cleanup_stale(self):
        mgr = TrackManager(max_stale_frames=10)
        mgr.get_or_create(1, 100)
        mgr.get_or_create(2, 200)
        cleaned = mgr.cleanup_stale(200)
        assert cleaned == 1  # track 1 被清理
        assert 2 in mgr.active_tracks

    def test_len(self):
        mgr = TrackManager()
        mgr.get_or_create(1, 100)
        mgr.get_or_create(2, 100)
        assert len(mgr) == 2

    def test_concurrent_access(self):
        """多线程并发访问 TrackManager 不应崩溃。"""
        import threading

        mgr = TrackManager()
        errors = []

        def writer():
            try:
                for i in range(100):
                    mgr.get_or_create(i % 10, i)
                    if i % 3 == 0:
                        mgr.mark_pending(i % 10)
                    if i % 5 == 0:
                        mgr.cancel_pending(i % 10)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    mgr.needs_recognition(i % 10)
                    mgr.get(i % 10)
                    _ = mgr.active_tracks
                    _ = len(mgr)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"并发访问出错: {errors}"


# ══════════════════════════════════════════════
#  AgentInference 单元测试（不调用 API）
# ══════════════════════════════════════════════

class TestAgentInferenceHelpers:
    def test_parse_response_json(self):
        content = '{"hull_number": "0014", "description": "白色大型客轮"}'
        result = AgentInference._parse_response(content)
        assert result["hull_number"] == "0014"
        assert result["description"] == "白色大型客轮"

    def test_parse_response_code_block(self):
        content = '```json\n{"hull_number": "0025", "description": "黑色散货船"}\n```'
        result = AgentInference._parse_response(content)
        assert result["hull_number"] == "0025"

    def test_parse_response_invalid_json(self):
        content = 'not json at all'
        result = AgentInference._parse_response(content)
        assert result["hull_number"] == ""
        assert result["description"] == "not json at all"

    def test_encode_image(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        b64 = AgentInference._encode_image(img)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_prompt_mode_switch(self):
        config = {
            "llm": {"model": "test", "api_key": "test", "base_url": "http://localhost:1/v1", "temperature": 0},
            "embed": {"model": "test", "api_key": "test", "base_url": "http://localhost:1/v1"},
            "retrieval": {"top_k": 3, "score_threshold": 0.5},
            "vector_store": {"persist_path": "/tmp/test_vs", "auto_rebuild": False},
            "app": {"log_level": "INFO", "ship_db_path": "/tmp/test.csv"},
        }
        agent = AgentInference(config=config, prompt_mode="detailed")
        assert agent.prompt_mode == "detailed"
        agent.set_prompt_mode("brief")
        assert agent.prompt_mode == "brief"

    def test_prompt_mode_invalid(self):
        config = {
            "llm": {"model": "test", "api_key": "test", "base_url": "http://localhost:1/v1", "temperature": 0},
            "embed": {"model": "test", "api_key": "test", "base_url": "http://localhost:1/v1"},
            "retrieval": {"top_k": 3, "score_threshold": 0.5},
            "vector_store": {"persist_path": "/tmp/test_vs", "auto_rebuild": False},
            "app": {"log_level": "INFO", "ship_db_path": "/tmp/test.csv"},
        }
        agent = AgentInference(config=config)
        with pytest.raises(ValueError):
            agent.set_prompt_mode("invalid")
