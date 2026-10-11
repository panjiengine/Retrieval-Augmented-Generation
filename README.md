# RAG + MCP Server

Template **Retrieval-Augmented Generation (RAG)** dari dokumen PDF yang
diekspos sebagai **MCP server**. Bisa dipakai lokal (stdio) maupun publik
via **ngrok** (Streamable HTTP) agar bisa diakses dari mana saja.

Pipeline: PDF → ekstrak teks → chunk → embeddings → index **Qdrant**
(cosine similarity, mode lokal persisten tanpa server terpisah).

## Prasyarat

- [uv](https://docs.astral.sh/uv/) (Python >= 3.14)
- Server embedding yang kompatibel OpenAI API, contoh `llama-server`:
  ```
  llama-server -m <model-embedding.gguf> --port 8080
  ```
- `ngrok` + akun (hanya untuk mode publik)

## Instalasi

```bash
uv sync
cp .env.example .env   # isi NGROK_AUTHTOKEN untuk mode publik
```

Taruh PDF sumber di `jurnal/` (atau di mana saja, lalu arahkan via `PDF_PATH`).

## Menjalankan MCP Server

Lokal (stdio — untuk Claude Desktop / opencode):

```bash
uv run python -m retrieval_augmented_generation.mcp_server
```

HTTP + ngrok (publik, satu perintah):

```bash
export NGROK_AUTHTOKEN="..."
uv run python run_public.py --port 8000 --domain xxxx.ngrok-free.app
```

Atau manual dua terminal:

```bash
MCP_PUBLIC=1 NGROK_DOMAIN=xxxx.ngrok-free.app \
  uv run python -m retrieval_augmented_generation.mcp_server --http --port 8000
ngrok http 8000 --url https://xxxx.ngrok-free.app
```

Endpoint MCP ada di `<url-ngrok>/mcp`, contoh config client
(`mcp.json.example`):

```json
{"type": "streamable", "url": "https://xxxx.ngrok-free.app/mcp"}
```

## Tools

| Tool | Fungsi |
|---|---|
| `rag_search(query, k=5)` | Cari potongan dokumen paling relevan (teks, halaman, skor) |
| `rag_stats()` | Statistik index: halaman, chunks, dimensi embedding |
| `rag_health()` | Cek file PDF + koneksi embedding API |
| `rag_rebuild()` | Bangun ulang index dari PDF |

## Konfigurasi (env)

| Variabel | Default | Keterangan |
|---|---|---|
| `PDF_PATH` | `jurnal/electronics-15-04648.pdf` | PDF sumber RAG |
| `EMBED_API_URL` | `http://127.0.0.1:8080/v1/embeddings` | Endpoint embedding |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Ukuran chunk & overlap karakter |
| `EMBED_BATCH` | `16` | Ukuran batch request embedding |
| `RAG_CACHE_DIR` | `.rag_cache` | Lokasi cache chunks + embeddings |
| `QDRANT_PATH` | `.qdrant` | Storage Qdrant lokal (abaikan jika `QDRANT_URL` diisi) |
| `QDRANT_URL` | — | URL server Qdrant, contoh `http://localhost:6333` (opsional) |
| `QDRANT_COLLECTION` | `rag-docs` | Nama collection |
| `MCP_PORT` / `MCP_PATH` | `8000` / `/mcp` | Port & path HTTP |
| `NGROK_AUTHTOKEN` / `NGROK_DOMAIN` | — | Token & domain reservasi ngrok |

Hasil embedding di-cache di `.rag_cache/` (berdasarkan hash path + mtime PDF),
jadi startup berikutnya instan selama PDF tidak berubah.

## Struktur

```
jurnal/                          PDF sumber
src/retrieval_augmented_generation/
  rag_store.py                   PDF → chunks → embeddings → FAISS
  mcp_server.py                  MCP server (stdio / streamable-http)
run_public.py                    sekali jalan: MCP HTTP + tunnel ngrok
mcp.json.example                 contoh config client
```
