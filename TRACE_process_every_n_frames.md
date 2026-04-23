# `process_every_n_frames` 参数代码链路

## 全链路总览

```
config.yaml (声明)
    │
    ▼
config.py (默认值字典)
    │
    ▼
pipeline/pipeline.py::__init__  (读取 + 防御)
    │
    ▼
pipeline/cli.py  (CLI 参数覆盖，可选)
    │
    ▼
pipeline/pipeline.py::run  (主循环判定)
    │
    ├── True  → Agent 推理 + 截图保存
    └── False → 跳过推理，仅 YOLO 检测 + 跟踪
```

---

## 第 1 层：配置声明

**文件：`config.yaml` 第 48 行**

```yaml
pipeline:
  # 每 N 帧处理一次（1 = 每帧都处理）
  process_every_n_frames: 30
```

> 这是用户可编辑的入口，设为 30 表示每 30 帧触发一次 Agent 推理。

---

## 第 2 层：默认值字典

**文件：`config.py` 第 41 行**

```python
"pipeline": {
    "concurrent_mode": False,
    "max_concurrent": 4,
    "max_queued_frames": 30,
    "process_every_n_frames": 30,    # ← 默认值，与 yaml 一致
    "output_dir": "./output",
    ...
}
```

> `load_config()` 会把 yaml 中的值覆盖到这里，所以最终取 yaml 的值。
> 这里写 30 是兜底，即使 yaml 缺失也能保证有值。

---

## 第 3 层：Pipeline 构造函数读取

**文件：`pipeline/pipeline.py` 第 62 行**

```python
self._process_every_n: int = max(1, pipe_cfg.get("process_every_n_frames", 1))
```

**拆解：**
```
pipe_cfg.get("process_every_n_frames", 1)  →  从配置取值，缺省为 1
max(1, ...)                                 →  防御：最小为 1，不会出现 0 或负数
self._process_every_n                       →  实例变量，值 = 30
```

---

## 第 4 层：CLI 参数覆盖（可选）

**文件：`pipeline/cli.py` 第 91 行 — 参数定义**

```python
parser.add_argument(
    "--process-every",
    type=int,
    default=1,
    help="每 N 帧处理一次（默认 1 = 每帧都处理）",
)
```

**文件：`pipeline/cli.py` 第 182 行 — 覆盖到 config**

```python
config["pipeline"]["process_every_n_frames"] = args.process_every
```

> **注意**：CLI 的 `default=1` 意味着不传 `--process-every` 时会把 config 覆盖为 1！
> 如果只想用 config.yaml 的 30，要么不走 CLI 覆盖逻辑，要么显式传 `--process-every 30`。

---

## 第 5 层：主循环触发判定

**文件：`pipeline/pipeline.py` 第 567-640 行 — 主循环**

```python
frame_id = 0                          # 第 567 行：初始化

while True:
    ret, frame = input_src.read()
    if not ret:
        break

    frame_id += 1                     # 第 599 行：先自增（从 1 开始）

    # ── YOLO 检测 + 跟踪（每帧都执行）──
    detections = self._detector.detect(frame, frame_id)
    for det in detections:
        self._tracker.get_or_create(det.track_id, frame_id)

    # ── process_every_n_frames 控制推理频率 ──
    should_process = (frame_id % self._process_every_n == 0)   # 第 628 行

    if should_process:                                           # 第 630 行
        # Agent 推理
        if self._concurrent_mode:
            self._concurrent_process(detections, frame_id)
        else:
            self._cascade_process(detections, frame_id)

    # ── 每 N 帧：有已识别的 track 就保存截图 ──
    if should_process:                                           # 第 638 行
        has_recognized = any(
            t.hull_number for t in self._tracker.active_tracks.values()
            if t.recognized
        )
        if has_recognized:
            self._saver.save(display_frame, frame_id)
```

---

## 执行时序（process_every_n_frames = 30）

```
frame_id:  1   2   3  ...  29  30  31  ...  59  60  61  ...  89  90
           │   │   │       │   │   │       │   │   │       │   │
YOLO+Track ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ← 每帧
Agent推理  ✗   ✗   ✗   ✗   ✗   ✓   ✗   ✗   ✗   ✓   ✗   ✗   ✗   ✓   ← 每30帧
截图保存   ✗   ✗   ✗   ✗   ✗   ✓*  ✗   ✗   ✗   ✓*  ✗   ✗   ✗   ✓*  ← 每30帧且有识别结果

✓* = 需要有已识别的船舶 track 才保存
```

---

## ⚠️ 潜在问题：CLI 默认值覆盖

**文件：`pipeline/cli.py` 第 91 行**

```python
"--process-every", type=int, default=1   # ← 默认是 1，不是 30
```

**第 182 行：**

```python
config["pipeline"]["process_every_n_frames"] = args.process_every
# 不传 --process-every 时，args.process_every = 1
# 会把 config.yaml 的 30 覆盖成 1！
```

**修复建议**：把 CLI 默认值改为 `None`，仅在用户显式传参时才覆盖：

```python
parser.add_argument(
    "--process-every",
    type=int,
    default=None,          # ← 改为 None
    help="每 N 帧处理一次（默认 1 = 每帧都处理）",
)
```

```python
# 第 182 行改为条件覆盖
if args.process_every is not None:
    config["pipeline"]["process_every_n_frames"] = args.process_every
```
