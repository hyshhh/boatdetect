"""船弦号数据库 — CSV 数据源 + FAISS 向量库 + 自动变更检测"""

from __future__ import annotations

import csv
import hashlib
import logging
from pathlib import Path
from typing import Any, Mapping

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import load_config

logger = logging.getLogger(__name__)

# ── 内置默认 CSV 内容 ──────────────────────────

DEFAULT_CSV_CONTENT = """hull_number,description
0014,白色大型客轮，上层建筑为蓝色涂装，船尾有直升机停机坪
0025,黑色散货船，船体有红色水线，甲板上配有龙门吊
0123,白色邮轮，船身有红蓝条纹装饰，三座烟囱
0256,灰色军舰，隐身外形设计，舰首配有垂直发射系统
0389,红色渔船，船身有白色编号，甲板配有拖网绞车
0455,绿色集装箱船，船体涂有大型LOGO，配有四台岸桥吊
0512,黄色挖泥船，船体宽大，中部有大型绞吸臂
0678,蓝色油轮，双壳结构，船尾有大型舵机舱
0789,白色科考船，船尾有A型吊架，甲板有多个实验室舱
"""

HASH_FILE_NAME = ".db_hash"


class DashScopeEmbeddings(Embeddings):
    """DashScope Embedding 封装，直接调用 OpenAI 兼容模式 API。"""

    def __init__(self, model: str, api_key: str, base_url: str):
        if not api_key or api_key.startswith("your-"):
            raise ValueError(
                "Embedding API Key 未配置。请在 config.yaml 中设置 embed.api_key，"
                "或在 .env 中设置 EMBED_API_KEY。"
            )
        self.model = model
        self.api_key = api_key
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import httpx
        import time

        max_retries = 3
        batch_size = 10  # DashScope text-embedding-v4 单次上限
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            last_error: Exception | None = None

            for attempt in range(max_retries):
                try:
                    payload = {"model": self.model, "input": batch}
                    resp = httpx.post(
                        self._url,
                        headers=self._headers,
                        json=payload,
                        timeout=60,
                    )
                    if resp.status_code == 429:
                        retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                        logger.warning("Embedding API 限流，%ds 后重试 (%d/%d)", retry_after, attempt + 1, max_retries)
                        time.sleep(retry_after)
                        continue
                    if resp.status_code >= 500:
                        logger.warning("Embedding API 服务错误 [%d]，%ds 后重试 (%d/%d)", resp.status_code, 2 ** attempt, attempt + 1, max_retries)
                        time.sleep(2 ** attempt)
                        continue
                    if not resp.is_success:
                        try:
                            err_body = resp.json()
                            err_msg = err_body.get("error", {}).get("message", resp.text[:300])
                        except Exception:
                            err_msg = resp.text[:300]
                        raise RuntimeError(
                            f"Embedding API 返回 {resp.status_code}: {err_msg}\n"
                            f"请检查 config.yaml 中 embed 配置（model / api_key / base_url）。"
                        )
                    data = resp.json()
                    batch_embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning("Embedding API 网络错误: %s，%ds 后重试 (%d/%d)", e, wait, attempt + 1, max_retries)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Embedding API 调用失败，已重试 {max_retries} 次") from last_error

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ShipDatabase:
    """
    船弦号数据库 — 双通道检索：
      1. 精确查找（dict，O(1)）
      2. FAISS 向量语义检索（RAG）

    数据源：CSV 文件（hull_number, description）
    自动变更检测：通过 MD5 哈希比对，CSV 变更时自动重建向量库。
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        db_path: str | None = None,
    ):
        if config is None:
            config = load_config()

        self._config = config

        embed_cfg = config.get("embed", {})
        retrieval_cfg = config.get("retrieval", {})
        vs_cfg = config.get("vector_store", {})

        # ── 确定 CSV 路径 ──
        csv_path = db_path or config.get("app", {}).get("ship_db_path")
        self._csv_path = self._resolve_csv_path(csv_path)

        # ── 加载数据 ──
        self._data = self._load_csv(self._csv_path)

        # ── Embedding 客户端 ──
        self._embeddings = DashScopeEmbeddings(
            model=embed_cfg.get("model", "Qwen3-Embedding-0.6B"),
            api_key=embed_cfg.get("api_key", ""),
            base_url=embed_cfg.get("base_url", "http://localhost:7891/v1"),
        )

        # ── 检索参数 ──
        self._top_k = retrieval_cfg.get("top_k", 3)
        self._score_threshold = retrieval_cfg.get("score_threshold", 0.5)

        # ── 向量库配置 ──
        self._persist_path = vs_cfg.get("persist_path", "./vector_store")
        self._auto_rebuild = vs_cfg.get("auto_rebuild", False)

        # ── 向量库（懒加载） ──
        self._vector_store: FAISS | None = None

    # ── CSV 路径解析 ────────────────────────────

    def _resolve_csv_path(self, db_path: str | None) -> Path:
        if db_path:
            p = Path(db_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"CSV 数据库文件不存在: {p}\n"
                    f"请检查 config.yaml 中 app.ship_db_path 配置。"
                )
        else:
            p = Path(__file__).resolve().parent.parent / "data" / "ships.csv"
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(DEFAULT_CSV_CONTENT.strip(), encoding="utf-8")
                logger.info("已创建默认 CSV 数据库: %s", p)

        return p.resolve()

    # ── CSV 加载 ────────────────────────────────

    @staticmethod
    def _load_csv(path: Path) -> dict[str, str]:
        data: dict[str, str] = {}
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                logger.error("CSV 文件为空或无法解析表头: %s", path)
                return data
            if "hull_number" not in reader.fieldnames:
                logger.error("CSV 文件缺少 hull_number 列，实际列: %s", reader.fieldnames)
                return data
            for row in reader:
                hn = (row.get("hull_number") or "").strip()
                desc = (row.get("description") or "").strip()
                if hn:
                    data[hn] = desc
        logger.info("从 CSV 加载了 %d 条船记录: %s", len(data), path)
        return data

    # ── 变更检测 ────────────────────────────────

    def _compute_csv_hash(self) -> str:
        return hashlib.md5(self._csv_path.read_bytes()).hexdigest()

    def _load_saved_hash(self) -> str | None:
        hash_file = Path(self._persist_path) / HASH_FILE_NAME
        if hash_file.exists():
            return hash_file.read_text(encoding="utf-8").strip()
        return None

    def _save_hash(self, csv_hash: str) -> None:
        persist_dir = Path(self._persist_path)
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / HASH_FILE_NAME).write_text(csv_hash, encoding="utf-8")

    def _csv_changed(self) -> bool:
        current_hash = self._compute_csv_hash()
        saved_hash = self._load_saved_hash()
        changed = current_hash != saved_hash
        if changed:
            logger.info("CSV 数据变更检测: 文件已修改，将重建向量库")
        return changed

    # ── 向量库构建 ─────────────────────────────

    def _build_documents(self) -> list[Document]:
        docs = []
        for hn, desc in self._data.items():
            content = f"弦号 {hn}\n{desc}"
            docs.append(Document(
                page_content=content,
                metadata={"hull_number": hn, "description": desc},
            ))
        return docs

    def _load_or_build_vector_store(self) -> FAISS:
        persist_dir = Path(self._persist_path)
        index_file = persist_dir / "index.faiss"

        csv_changed = self._csv_changed()

        if not self._auto_rebuild and not csv_changed and index_file.exists():
            try:
                logger.info("从 %s 加载向量库缓存…", persist_dir)
                vs = FAISS.load_local(
                    str(persist_dir),
                    self._embeddings,
                    allow_dangerous_deserialization=True,
                )
                logger.info("向量库缓存加载成功")
                return vs
            except Exception as e:
                logger.warning("缓存加载失败（%s），将重新构建", e)

        # CSV 变化时，先重新加载数据
        if csv_changed:
            logger.info("重新加载 CSV 数据: %s", self._csv_path)
            self._data = self._load_csv(self._csv_path)

        docs = self._build_documents()
        logger.info("正在构建 FAISS 向量库（%d 条文档）…", len(docs))
        vs = FAISS.from_documents(docs, self._embeddings)

        persist_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(persist_dir))

        self._save_hash(self._compute_csv_hash())
        logger.info("向量库已持久化到 %s，哈希已更新", persist_dir)

        return vs

    @property
    def vector_store(self) -> FAISS:
        if self._vector_store is None or self._csv_changed():
            self._vector_store = self._load_or_build_vector_store()
        return self._vector_store

    # ── 精确查找 ──────────────────────────────

    def lookup(self, hull_number: str) -> str | None:
        return self._data.get(hull_number.strip())

    # ── 语义检索 ──────────────────────────────

    def semantic_search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self._top_k
        results_with_score = self.vector_store.similarity_search_with_score(query, k=k)

        results = []
        for doc, distance in results_with_score:
            score = float(1.0 / (1.0 + distance))
            results.append({
                "hull_number": doc.metadata["hull_number"],
                "description": doc.metadata["description"],
                "score": round(score, 4),
            })
        return results

    def semantic_search_filtered(self, query: str) -> list[dict]:
        results = self.semantic_search(query, top_k=self._top_k)
        return [r for r in results if r["score"] >= self._score_threshold]

    # ── 属性 ──────────────────────────────────

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    @property
    def hull_numbers(self) -> list[str]:
        return list(self._data.keys())

    @property
    def descriptions(self) -> list[str]:
        return list(self._data.values())

    @property
    def items(self) -> Mapping[str, str]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)
