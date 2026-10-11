# Retrieval-Augmented Generation (RAG) + MCP Server

RAG dari jurnal elektronika (`electronics-15-04648.pdf`) yang diekspos sebagai
**MCP server**, bisa dipakai lokal (stdio) maupun publik via **ngrok**
(Streamable HTTP) agar semua orang bisa akses.

Pipeline: PDF → ekstrak teks (`pymupdf`) → chunk (1000/150) →
embeddings via `llama-server` (granite-embedding) → index **FAISS**
(cosine similarity via inner product ternormalisasi).

## Prasyarat

- [uv](https://docs.astral.sh/uv/) (Python >= 3.14)
- `llama-server` berjalan dengan model embedding di port 8080:
  ```
  llama-server -m models/granite-embedding-107m-multilingual-Q4_K_M.gguf --port 8080
  ```
- `ngrok` + akun (hanya untuk mode publik)

## Instalasi

```bash
uv sync
cp .env.example .env   # isi NGROK_AUTHTOKEN untuk mode publik
```

## Menjalankan MCP Server

Lokal (stdio — untuk Claude Desktop / opencode):

```bash
uv run python -m retrieval_augmented_generation.mcp_server
```

HTTP + ngrok (publik, satu perintah):

```bash
export NGROK_AUTHTOKEN="..."
uv run python run_public.py --port 8000 --domain alive-set-crappie.ngrok-free.app
```

Atau manual dua terminal:

```bash
MCP_PUBLIC=1 NGROK_DOMAIN=alive-set-crappie.ngrok-free.app \
  uv run python -m retrieval_augmented_generation.mcp_server --http --port 8000
ngrok http 8000 --url https://alive-set-crappie.ngrok-free.app
```

Endpoint publik: `https://alive-set-crappie.ngrok-free.app/mcp`

Contoh config client (`mcp.json.example`):

```json
{"type": "streamable", "url": "https://alive-set-crappie.ngrok-free.app/mcp"}
```

## Tools

| Tool | Fungsi |
|---|---|
| `rag_search(query, k=5)` | Cari potongan jurnal paling relevan (teks, halaman, skor) |
| `rag_stats()` | Statistik index: halaman, chunks, dimensi embedding |
| `rag_health()` | Cek PDF + koneksi embedding API |
| `rag_rebuild()` | Bangun ulang index dari PDF |

## Konfigurasi (env)

| Variabel | Default | Keterangan |
|---|---|---|
| `PDF_PATH` | `jurnal/electronics-15-04648.pdf` | PDF sumber RAG |
| `EMBED_API_URL` | `http://127.0.0.1:8080/v1/embeddings` | Endpoint embedding llama-server |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Ukuran chunk & overlap karakter |
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
