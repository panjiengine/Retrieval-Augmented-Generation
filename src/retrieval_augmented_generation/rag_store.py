"""Core RAG logic: PDF -> chunks -> embeddings (llama-server) -> FAISS.

Dipakai oleh MCP server. Diadaptasi dari rag-document.ipynb.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import faiss
import numpy as np
import pymupdf
import requests

# --- Konfigurasi via env ---
PROJECT_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path.cwd()

def _default_pdf() -> str:
    candidates = [
        PROJECT_ROOT / "jurnal" / "electronics-15-04648.pdf",
        Path(__file__).parent / "electronics-15-04648.pdf",
        PROJECT_ROOT / "src" / "retrieval_augmented_generation" / "electronics-15-04648.pdf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])

PDF_PATH = os.getenv("PDF_PATH", _default_pdf())
EMBED_API_URL = os.getenv("EMBED_API_URL", "http://127.0.0.1:8080/v1/embeddings")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "16"))
CACHE_DIR = Path(os.getenv("RAG_CACHE_DIR", str(PROJECT_ROOT / ".rag_cache")))


def make_chunks(pages: list[dict], chunk_size: int = 1000, overlap: int = 150) -> list[dict]:
    chunks: list[dict] = []
    for item in pages:
        text = item["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"page": item["page"], "text": chunk})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def embed_texts(texts: list[str], api_url: str = EMBED_API_URL, batch_size: int = EMBED_BATCH) -> np.ndarray:
    all_vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = requests.post(api_url, json={"input": batch}, timeout=180)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda item: item["index"])
        all_vectors.extend(item["embedding"] for item in data)
    return np.asarray(all_vectors, dtype=np.float32)


def _build_faiss(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    return index


class RagStore:
    """Singleton store: load PDF sekali, simpan FAISS index di memori + cache disk."""

    _instance: "RagStore | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.pdf_path = PDF_PATH
        self.api_url = EMBED_API_URL
        self.chunks: list[dict] = []
        self.index: faiss.IndexFlatIP | None = None
        self.dimension: int = 0
        self.num_pages: int = 0
        self._built = False
        self._build_lock = threading.Lock()

    @classmethod
    def get(cls) -> "RagStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- cache helpers --
    def _cache_key(self) -> str:
        p = Path(self.pdf_path)
        mtime = str(p.stat().st_mtime) if p.exists() else "missing"
        raw = f"{p.resolve()}|{mtime}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|{self.api_url}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _cache_files(self) -> tuple[Path, Path, Path, Path]:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key = self._cache_key()
        return (
            CACHE_DIR / f"chunks-{key}.json",
            CACHE_DIR / f"embeddings-{key}.npy",
            CACHE_DIR / f"meta-{key}.json",
            CACHE_DIR / f"faiss-{key}.index",
        )

    def build(self, force: bool = False) -> dict:
        with self._build_lock:
            if self._built and not force:
                return self.stats()
            chunk_file, emb_file, meta_file, index_file = self._cache_files()

            embeddings: np.ndarray | None = None
            if not force and chunk_file.exists() and emb_file.exists() and meta_file.exists():
                try:
                    self.chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
                    embeddings = np.load(str(emb_file))
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self.dimension = int(embeddings.shape[1])
                    self.num_pages = int(meta.get("num_pages", 0))
                    self.pdf_path = meta.get("pdf_path", self.pdf_path)
                except Exception:
                    embeddings = None  # cache rusak -> rebuild

            if embeddings is None:
                # --- rebuild dari PDF ---
                pdf = Path(self.pdf_path)
                if not pdf.exists():
                    raise FileNotFoundError(f"PDF tidak ditemukan: {self.pdf_path}")
                doc = pymupdf.open(str(pdf))
                pages: list[dict] = []
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append({"page": page_num, "text": text})
                doc.close()
                if not pages:
                    raise ValueError("PDF tidak mengandung teks yang bisa diekstrak.")

                self.num_pages = 0
                # total halaman pdf (bukan hanya yang berisi teks)
                with pymupdf.open(str(pdf)) as d:
                    self.num_pages = len(d)

                self.chunks = make_chunks(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
                texts = [c["text"] for c in self.chunks]
                embeddings = embed_texts(texts, api_url=self.api_url, batch_size=EMBED_BATCH)
                # simpan cache sebelum normalisasi? simpan mentah, normalisasi saat load
                np.save(str(emb_file), embeddings)
                chunk_file.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
                meta_file.write_text(
                    json.dumps({"num_pages": self.num_pages, "pdf_path": str(pdf), "dimension": int(embeddings.shape[1])}),
                    encoding="utf-8",
                )
                self.dimension = int(embeddings.shape[1])

            self.index = _build_faiss(embeddings)
            try:
                faiss.write_index(self.index, str(index_file))
            except Exception:
                pass
            self._built = True
            return self.stats()

    def ensure_built(self) -> None:
        if not self._built:
            self.build()

    def stats(self) -> dict:
        return {
            "pdf_path": str(self.pdf_path),
            "num_pages": self.num_pages,
            "num_chunks": len(self.chunks),
            "dimension": self.dimension,
            "indexed_vectors": int(self.index.ntotal) if self.index is not None else 0,
            "embed_api": self.api_url,
            "built": self._built,
        }

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query or not query.strip():
            raise ValueError("query tidak boleh kosong.")
        self.ensure_built()
        assert self.index is not None
        k = max(1, min(int(k), int(self.index.ntotal), 10))
        qv = embed_texts([query.strip()], api_url=self.api_url, batch_size=1)
        faiss.normalize_L2(qv)
        scores, ids = self.index.search(qv, k)
        results: list[dict] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self.chunks):
                continue
            item = self.chunks[idx]
            results.append({"text": item["text"], "page": item["page"], "score": float(score)})
        return results

    def health(self) -> dict:
        out: dict = {"rag_built": self._built}
        try:
            r = requests.get(self.api_url.replace("/v1/embeddings", "/v1/models"), timeout=5)
            out["embed_api_ok"] = r.status_code == 200
            out["embed_api_status"] = r.status_code
        except Exception as e:
            out["embed_api_ok"] = False
            out["embed_api_error"] = str(e)
        try:
            p = Path(self.pdf_path)
            out["pdf_exists"] = p.exists()
        except Exception:
            out["pdf_exists"] = False
        return out


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Bangun vector store RAG dari PDF")
    parser.add_argument("pdf", help='path PDF sumber, contoh: jurnal/new.pdf')
    parser.add_argument("--force", action="store_true", help="embed ulang dari nol, abaikan cache")
    parser.add_argument("--test", default="", help='uji retrieval setelah build, contoh: --test "fault location"')
    parser.add_argument("--api-url", default="", help="endpoint embedding (default dari EMBED_API_URL)")
    args = parser.parse_args()

    store = RagStore()
    store.pdf_path = args.pdf
    if args.api_url:
        store.api_url = args.api_url
    stats = store.build(force=args.force)
    print(f"SELESAI: {stats['num_chunks']} chunks, {stats['num_pages']} halaman, "
          f"dimensi {stats['dimension']} [{stats['pdf_path']}]", flush=True)

    if args.test:
        for i, r in enumerate(store.search(args.test, k=3), start=1):
            snippet = r["text"][:180].replace("\n", " ")
            print(f"  [{i}] skor={r['score']:.4f} hal.{r['page']}: {snippet}...", flush=True)


if __name__ == "__main__":
    _cli()
