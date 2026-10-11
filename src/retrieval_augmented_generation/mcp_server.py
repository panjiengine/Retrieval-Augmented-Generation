"""MCP Server untuk RAG jurnal elektronika.

Transport:
- lokal (stdio): untuk Claude Desktop / opencode
- publik (streamable-http): untuk di-share via ngrok agar semua orang bisa akses

Contoh run lokal stdio:
    uv run python -m retrieval_augmented_generation.mcp_server

Contoh run HTTP (untuk ngrok):
    uv run python -m retrieval_augmented_generation.mcp_server --http --port 8000
    ngrok http 8000
"""
from __future__ import annotations

import argparse
import os

try:
    from mcp.server.mcpserver import MCPServer as _MCPServer

    _IS_V2 = True
except ImportError:  # fallback mcp v1 (FastMCP)
    from mcp.server.fastmcp import FastMCP as _MCPServer  # type: ignore

    _IS_V2 = False

from .rag_store import RagStore

mcp = _MCPServer(
    name="rag-jurnal-elektronika",
    title="RAG Jurnal Elektronika",
    description="Retrieval-Augmented Generation dari jurnal electronics-15-04648.pdf via Qdrant + granite embeddings.",
    instructions="Gunakan rag_search untuk mencari potongan jurnal yang relevan sebelum menjawab pertanyaan elektronika.",
)


@mcp.tool()
def rag_search(query: str, k: int = 5) -> list[dict]:
    """Cari potongan jurnal paling relevan secara semantik.

    Args:
        query: pertanyaan atau kata kunci (contoh: 'fault location traveling wave').
        k: jumlah hasil (1-10, default 5).
    """
    store = RagStore.get()
    return store.search(query, k=k)


@mcp.tool()
def rag_stats() -> dict:
    """Kembalikan statistik index RAG: jumlah halaman, chunks, dimensi embedding."""
    store = RagStore.get()
    store.ensure_built()
    return store.stats()


@mcp.tool()
def rag_health() -> dict:
    """Cek kesehatan: apakah PDF ada dan embedding API (llama-server) reachable."""
    return RagStore.get().health()


@mcp.tool()
def rag_rebuild() -> dict:
    """Bangun ulang index dari PDF (paksa re-embed). Gunakan jika PDF diganti."""
    store = RagStore.get()
    return store.build(force=True)


def _build_index_on_startup() -> None:
    # Build saat startup agar tool pertama tidak timeout.
    # Jika gagal (mis. llama-server mati), biarkan server tetap jalan;
    # tool akan me-return error yang jelas saat dipanggil.
    try:
        print("[rag] building index...", flush=True)
        stats = RagStore.get().build()
        print(f"[rag] ready: {stats['num_chunks']} chunks, {stats['num_pages']} pages.", flush=True)
    except Exception as e:
        print(f"[rag] WARNING: gagal build saat startup: {e}", flush=True)
        print("[rag] Server tetap jalan. Pastikan llama-server aktif di EMBED_API_URL.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Server RAG")
    parser.add_argument("--http", action="store_true", help="jalankan sebagai Streamable HTTP (untuk ngrok)")
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")))
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0" if os.getenv("MCP_PUBLIC") else "127.0.0.1"))
    parser.add_argument("--path", default=os.getenv("MCP_PATH", "/mcp"))
    parser.add_argument("--no-prebuild", action="store_true", help="jangan build index saat startup")
    args = parser.parse_args()

    if not args.no_prebuild:
        _build_index_on_startup()

    if args.http or os.getenv("MCP_TRANSPORT") == "streamable-http":
        host = args.host
        # Kalau di-share publik, bind ke 0.0.0.0
        if os.getenv("MCP_PUBLIC"):
            host = "0.0.0.0"
        print(f"[mcp] Streamable HTTP di http://{host}:{args.port}{args.path}", flush=True)
        if _IS_V2:
            kwargs: dict = dict(
                host=host,
                port=args.port,
                streamable_http_path=args.path,
            )
            # Izinkan Host header dari domain ngrok (DNS rebinding protection).
            # Contoh: NGROK_DOMAIN=alive-set-crappie.ngrok-free.app
            # atau MCP_ALLOWED_HOSTS="alive-set-crappie.ngrok-free.app,*.ngrok-free.app"
            allowed: list[str] = ["127.0.0.1", "localhost"]
            for env_key in ("NGROK_DOMAIN", "MCP_ALLOWED_HOSTS"):
                raw = os.getenv(env_key, "")
                if raw:
                    allowed.extend(h.strip() for h in raw.split(",") if h.strip())
            # default: izinkan semua subdomain ngrok-free agar domain random tetap bisa
            if os.getenv("MCP_PUBLIC"):
                allowed.append("*.ngrok-free.app")
            # dedup, pertahankan urutan
            allowed = list(dict.fromkeys(allowed))
            if allowed:
                try:
                    from mcp.server.transport_security import TransportSecuritySettings

                    origins = [
                        f"https://{h.replace('*.', '')}" for h in allowed if not h.startswith("http") and "*" not in h
                    ]
                    kwargs["transport_security"] = TransportSecuritySettings(
                        allowed_hosts=allowed,
                        allowed_origins=origins,
                    )
                    print(f"[mcp] allowed_hosts: {allowed}", flush=True)
                except ImportError:
                    pass
            mcp.run(transport="streamable-http", **kwargs)
        else:
            # mcp v1: FastMCP.run menerima host/port/path juga
            mcp.run(transport="streamable-http", host=host, port=args.port, path=args.path)  # type: ignore
    else:
        print("[mcp] stdio mode (untuk Claude Desktop / opencode lokal)", flush=True)
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
