"""
自动构建数据库脚本 — 扫描图片目录，调用视觉模型识别船只，自动生成弦号和描述并存入 CSV。

用法：
    python3 build_db.py <图片目录路径>

流程：
    1. 扫描目录中所有图片
    2. 对每张图片调用视觉模型识别船只
    3. 生成弦号和描述
    4. 检查弦号是否已存在（所有场景均查重）
    5. 询问用户确认弦号是否正确
       - 按 1：确认 / 跳过
       - 按 2：覆盖 / 手动输入
    6. 立即写入 CSV，继续下一张
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.prompt import Prompt

from config import load_config

logger = logging.getLogger(__name__)
console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

RECOGNITION_PROMPT = """请仔细观察这张图片，识别图中的船只。

请以 JSON 格式返回，包含以下字段：
{
  "hull_number": "船身上的弦号文字（如果看不清则留空字符串）",
  "description": "对船只外观的详细中文描述，包括：船体颜色、船型（客轮/货轮/军舰/渔船等）、大小特征、上层建筑特征、特殊标志等"
}

只返回 JSON，不要其他内容。"""


def _encode_image(image_path: Path) -> str:
    """将图片编码为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_mime_type(image_path: Path) -> str:
    """根据扩展名获取 MIME 类型。"""
    ext = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


def recognize_ship(image_path: Path, llm: ChatOpenAI) -> dict:
    """
    调用视觉模型识别图片中的船只。
    返回 {"hull_number": str, "description": str}
    """
    b64 = _encode_image(image_path)
    mime = _get_mime_type(image_path)

    msg = HumanMessage(content=[
        {"type": "text", "text": RECOGNITION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ])

    resp = llm.invoke([msg])
    content = resp.content.strip()

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
                logger.warning("模型返回非 JSON 格式，使用原始文本: %s", content[:200])
                result = {"hull_number": "", "description": content}
        else:
            logger.warning("模型返回非 JSON 格式，使用原始文本: %s", content[:200])
            result = {"hull_number": "", "description": content}

    return {
        "hull_number": str(result.get("hull_number", "")).strip(),
        "description": str(result.get("description", "")).strip(),
    }


def load_existing_csv(csv_path: Path) -> dict[str, str]:
    """加载已有 CSV 数据，返回 {hull_number: description}。"""
    data: dict[str, str] = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hn = row.get("hull_number", "").strip()
                desc = row.get("description", "").strip()
                if hn:
                    data[hn] = desc
    return data


def confirm_hull_number(detected: str) -> str:
    """
    询问用户确认弦号。
    返回最终确认的弦号。
    """
    if detected:
        console.print(f"\n  识别到弦号: [bold cyan]{detected}[/bold cyan]")
        console.print("  按 [bold green]1[/bold green] 确认，按 [bold yellow]2[/bold yellow] 手动输入正确弦号")
    else:
        console.print("\n  [dim]未识别到弦号[/dim]")
        console.print("  按 [bold green]1[/bold green] 跳过弦号（仅保存描述），按 [bold yellow]2[/bold yellow] 手动输入弦号")

    while True:
        try:
            choice = Prompt.ask("  请选择", choices=["1", "2"], default="1")
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [yellow]已取消[/yellow]")
            return detected
        if choice == "1":
            return detected
        elif choice == "2":
            try:
                manual = Prompt.ask("  请输入正确弦号").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [yellow]已取消，使用原弦号[/yellow]")
                return detected
            if manual:
                return manual
            console.print("  [red]弦号不能为空，请重新选择[/red]")


def scan_images(directory: Path) -> list[Path]:
    """扫描目录中所有支持的图片文件，返回排序后的列表。"""
    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    # 去重 + 排序
    images = sorted(set(images), key=lambda p: p.name.lower())
    return images


def main() -> None:
    """主入口。"""
    # 参数解析
    if len(sys.argv) < 2:
        console.print("用法: python3 build_db.py <图片目录路径>")
        console.print("示例: python3 build_db.py ./images")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        console.print(f"[red]错误: '{image_dir}' 不是有效的目录[/red]")
        sys.exit(1)

    # 加载配置
    config = load_config()
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model", "Qwen/Qwen3-VL-4B-AWQ")
    base_url = llm_cfg.get("base_url", "http://localhost:7890/v1")
    csv_path = Path(config.get("app", {}).get("ship_db_path") or "data/ships.csv")

    # 视觉模型客户端
    llm = ChatOpenAI(
        model=model,
        api_key=llm_cfg.get("api_key", "abc123"),
        base_url=base_url,
        temperature=0.0,
        max_tokens=1024,
    )

    # 确保 CSV 目录存在
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载已有数据
    existing = load_existing_csv(csv_path)
    console.print(f"\n📦 已有数据库: {csv_path}（{len(existing)} 条记录）")

    # 扫描图片
    images = scan_images(image_dir)
    if not images:
        console.print(f"在 '{image_dir}' 中未找到支持的图片文件")
        console.print(f"支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)

    console.print(f"🖼️  找到 {len(images)} 张图片，开始识别...\n")
    console.print(f"📡 使用模型: {model}")
    console.print(f"📡 服务地址: {base_url}\n")

    success_count = 0
    skip_count = 0

    for idx, image_path in enumerate(images, 1):
        console.print(f"{'='*60}")
        console.print(f"[{idx}/{len(images)}] 处理: {image_path.name}")
        console.print(f"{'='*60}")

        # 1. 调用模型识别
        try:
            result = recognize_ship(image_path, llm)
        except Exception as e:
            console.print(f"  ❌ 识别失败: [red]{e}[/red]")
            skip_count += 1
            continue

        hull_number = result["hull_number"]
        description = result["description"]

        console.print(f"\n  📝 识别结果:")
        console.print(f"     弦号: {hull_number or '(未识别)'}")
        console.print(f"     描述: {description}")

        # 2. 检查弦号是否已存在
        if hull_number and hull_number in existing:
            console.print(f"\n  ⚠️  弦号 [{hull_number}] 已存在于数据库中")
            console.print(f"     现有描述: {existing[hull_number]}")
            console.print("  按 [bold green]1[/bold green] 跳过（保留原记录）")
            console.print("  按 [bold yellow]2[/bold yellow] 覆盖为新描述")
            console.print("  按 [bold red]3[/bold red] 手动输入新弦号")

            while True:
                try:
                    choice = Prompt.ask("  请选择", choices=["1", "2", "3"])
                except (KeyboardInterrupt, EOFError):
                    console.print("\n  [yellow]已取消，跳过[/yellow]")
                    skip_count += 1
                    break
                if choice == "1":
                    console.print("  ⏭️  已跳过")
                    skip_count += 1
                    break
                elif choice == "2":
                    existing[hull_number] = description
                    _rewrite_csv(csv_path, existing)
                    console.print(f"  ✅ 已覆盖弦号 [{hull_number}]")
                    success_count += 1
                    break
                elif choice == "3":
                    hull_number = confirm_hull_number("")
                    if hull_number and hull_number in existing:
                        console.print(f"\n  ⚠️  手动弦号 [{hull_number}] 也已存在")
                        console.print(f"     现有描述: {existing[hull_number]}")
                        console.print("  按 [bold green]1[/bold green] 跳过（保留原记录）")
                        console.print("  按 [bold yellow]2[/bold yellow] 覆盖为新描述")
                        try:
                            sub_choice = Prompt.ask("  请选择", choices=["1", "2"], default="1")
                        except (KeyboardInterrupt, EOFError):
                            console.print("\n  [yellow]已取消，跳过[/yellow]")
                            skip_count += 1
                            break
                        if sub_choice == "1":
                            console.print("  ⏭️  已跳过")
                            skip_count += 1
                            break
                        else:
                            existing[hull_number] = description
                            _rewrite_csv(csv_path, existing)
                            console.print(f"  ✅ 已覆盖弦号 [{hull_number}]")
                            success_count += 1
                            break
                    elif hull_number:
                        existing[hull_number] = description
                        _rewrite_csv(csv_path, existing)
                        console.print(f"  ✅ 已保存弦号 [{hull_number}]")
                        success_count += 1
                        break
                    else:
                        console.print("  ⏭️  弦号为空，已跳过")
                        skip_count += 1
                        break
        else:
            # 3. 新弦号，确认是否正确
            hull_number = confirm_hull_number(hull_number)

            if hull_number:
                # 检查手动输入的弦号是否也已存在
                if hull_number in existing:
                    console.print(f"\n  ⚠️  弦号 [{hull_number}] 已存在")
                    console.print(f"     现有描述: {existing[hull_number]}")
                    console.print("  按 [bold green]1[/bold green] 跳过（保留原记录）")
                    console.print("  按 [bold yellow]2[/bold yellow] 覆盖为新描述")
                    try:
                        choice = Prompt.ask("  请选择", choices=["1", "2"], default="1")
                    except (KeyboardInterrupt, EOFError):
                        console.print("\n  [yellow]已取消，跳过[/yellow]")
                        skip_count += 1
                        choice = "1"
                    if choice == "1":
                        console.print("  ⏭️  已跳过")
                        skip_count += 1
                    else:
                        existing[hull_number] = description
                        _rewrite_csv(csv_path, existing)
                        console.print(f"  ✅ 已覆盖弦号 [{hull_number}]")
                        success_count += 1
                else:
                    existing[hull_number] = description
                    _rewrite_csv(csv_path, existing)
                    console.print(f"  ✅ 已保存弦号 [{hull_number}]")
                    success_count += 1
            else:
                # 无弦号，仅保存描述（用图片名作为键）
                fallback_key = image_path.stem
                if fallback_key in existing:
                    # 生成唯一 key: 文件名 + 序号
                    i = 2
                    while f"{fallback_key}_{i}" in existing:
                        i += 1
                    fallback_key = f"{fallback_key}_{i}"
                    console.print(f"\n  ⚠️  文件名已用作弦号，自动改为 [{fallback_key}]")

                existing[fallback_key] = description
                _rewrite_csv(csv_path, existing)
                console.print(f"  ✅ 已保存（使用 [{fallback_key}] 作为弦号）")
                success_count += 1

        console.print()

    # 汇总
    console.print(f"\n{'='*60}")
    console.print(f"📊 处理完成")
    console.print(f"   总计: {len(images)} 张图片")
    console.print(f"   成功: {success_count} 条")
    console.print(f"   跳过: {skip_count} 条")
    console.print(f"   数据库: {csv_path}（共 {len(existing)} 条记录）")
    console.print(f"{'='*60}")


def _rewrite_csv(csv_path: Path, data: dict[str, str]) -> bool:
    """原子写入 CSV 文件（先写临时文件再重命名，防止写入中断导致数据丢失）。
    返回 True 成功，False 失败。"""
    tmp_path = csv_path.with_suffix(".csv.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["hull_number", "description"])
            for hn, desc in data.items():
                writer.writerow([hn, desc])
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        console.print(f"  [red]⚠️  写入 CSV 失败: {e}[/red]")
        console.print("  [yellow]数据仍在内存中，下次成功写入时会自动保存[/yellow]")
        logger.error("写入 CSV 失败: %s", e)
        return False
    tmp_path.replace(csv_path)
    return True


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    main()
