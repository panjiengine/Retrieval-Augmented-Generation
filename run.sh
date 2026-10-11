#!/usr/bin/env bash
# Sekali jalan: Qdrant + llama-server (Qwen embedding) + MCP RAG + ngrok.
# Kalau sebuah layanan sudah jalan di portnya, di-skip (idempoten).
# Berhenti rapi dengan Ctrl+C.
set -euo pipefail

# --- Konfigurasi (ubah sesuai kebutuhan) ---
QDRANT_BIN="$HOME/qdrant/qdrant"
QDRANT_PORT=6333
QDRANT_URL="http://127.0.0.1:6333"
LLAMA_BIN="$HOME/llama.cpp/build-vulkan/bin/llama-server"
MODEL="$HOME/models/qwen3-embedding-0.6b-q4_k_m.gguf"
EMBED_PORT=8081
EMBED_API_URL="http://127.0.0.1:8081/v1/embeddings"
MCP_PORT=8000
PDF_PATH="jurnal/interpolation.pdf"
QDRANT_COLLECTION="rag-interpolation"
NGROK_DOMAIN="alive-set-crappie.ngrok-free.app"

cd "$(dirname "$0")"
mkdir -p logs
rm -f logs/.pids
PIDS=""

die() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' tidak ditemukan"; }

cleanup() {
  echo; echo "[run] mematikan layanan..."
  # shellcheck disable=SC2086
  kill $PIDS 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# tunggu http://host:port respons 2xx/3xx/4xx (maks ~60 dtk)
wait_http() { # url, nama
  for _ in $(seq 1 30); do
    code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" "$1" 2>/dev/null || echo 000)
    case "$code" in 2*|3*|4*) echo "[run] $2 OK ($1)"; return 0;; esac
    sleep 2
  done
  die "$2 tidak merespons di $1 (cek logs/)"
}

start_bg() { # nama, logfile, command...
  local nama="$1" log="$2"; shift 2
  echo "[run] menyalakan $nama..."
  "$@" >"logs/$log" 2>&1 &
  PIDS="$PIDS $!"
}

need curl
[[ -x "$QDRANT_BIN" ]] || die "qdrant tidak ada di $QDRANT_BIN"
[[ -x "$LLAMA_BIN" ]] || die "llama-server tidak ada di $LLAMA_BIN"
[[ -f "$MODEL" ]] || die "model tidak ada: $MODEL"
command -v ngrok >/dev/null || die "ngrok tidak ditemukan"
command -v uv >/dev/null || die "uv tidak ditemukan"

# 1. Qdrant
if curl -s -m 3 -o /dev/null "http://127.0.0.1:$QDRANT_PORT/"; then
  echo "[run] qdrant sudah jalan, skip"
else
  (cd "$(dirname "$QDRANT_BIN")" && ./qdrant >"$OLDPWD/logs/qdrant.log" 2>&1 & echo $! >>"$OLDPWD/logs/.pids")
  PIDS="$PIDS $(tail -n 1 logs/.pids)"
  wait_http "http://127.0.0.1:$QDRANT_PORT/" "qdrant"
fi

# 2. Model embedding
if curl -s -m 3 -o /dev/null "http://127.0.0.1:$EMBED_PORT/v1/models"; then
  echo "[run] llama-server sudah jalan, skip"
else
  start_bg "llama-server (qwen)" "embed.log" "$LLAMA_BIN" -m "$MODEL" --port "$EMBED_PORT" --embedding
  wait_http "http://127.0.0.1:$EMBED_PORT/v1/models" "llama-server"
fi

# 3. MCP server
if curl -s -m 3 -o /dev/null -X POST "http://127.0.0.1:$MCP_PORT/mcp" \
    -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
    -d '{"jsonrpc":"2.0","id":0,"method":"ping","params":{}}'; then
  echo "[run] mcp server sudah jalan, skip"
else
  echo "[run] menyalakan mcp server..."
  MCP_PUBLIC=1 NGROK_DOMAIN="$NGROK_DOMAIN" QDRANT_URL="$QDRANT_URL" \
  QDRANT_COLLECTION="$QDRANT_COLLECTION" EMBED_API_URL="$EMBED_API_URL" \
  PDF_PATH="$PDF_PATH" \
    uv run python -m retrieval_augmented_generation.mcp_server --http --port "$MCP_PORT" >logs/mcp.log 2>&1 &
  PIDS="$PIDS $!"
  sleep 12  # MCP butuh waktu build index saat startup
fi

# 4. ngrok
if curl -s -m 5 "https://$NGROK_DOMAIN/" 2>/dev/null | grep -q "ERR_NGROK_3200"; then
  start_bg "ngrok" "ngrok.log" ngrok http "$MCP_PORT" --url "https://$NGROK_DOMAIN" --log stdout
  sleep 8
else
  echo "[run] tunnel ngrok sudah online, skip"
fi

echo
echo "============================================================"
echo " SEMUA JALAN"
echo "  Qdrant      : http://127.0.0.1:$QDRANT_PORT"
echo "  Embedding   : http://127.0.0.1:$EMBED_PORT/v1"
echo "  MCP lokal   : http://127.0.0.1:$MCP_PORT/mcp"
echo "  MCP publik  : https://$NGROK_DOMAIN/mcp"
echo "  Config client: {\"type\": \"streamable\", \"url\": \"https://$NGROK_DOMAIN/mcp\"}"
echo "  Log         : logs/"
echo "  Berhenti    : Ctrl+C"
echo "============================================================"
wait
