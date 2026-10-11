"""Jalankan MCP RAG + expose publik via ngrok.

Cara pakai:
1. Daftar di https://dashboard.ngrok.com -> salin Authtoken.
2. export NGROK_AUTHTOKEN="xxx"   (atau isi file .env)
3. uv run python run_public.py --port 8000

Script ini akan:
- membuka tunnel ngrok -> localhost:PORT
- mencetak URL publik (tambahkan /mcp untuk endpoint MCP)
- menjalankan MCP server Streamable HTTP di 0.0.0.0:PORT

Orang lain tinggal pakai URL itu sebagai MCP server remote.

Contoh client config (opencode / Claude):
    {"type": "streamable", "url": "https://xxxx.ngrok-free.app/mcp"}
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _get_token(provided: str | None) -> str:
    token = provided or os.getenv("NGROK_AUTHTOKEN", "")
    if token:
        return token
    # fallback: biarkan ngrok pakai authtoken dari ~/.config/ngrok/ngrok.yml
    # (hasil `ngrok config add-authtoken ...`)
    cfg = os.path.expanduser("~/.config/ngrok/ngrok.yml")
    if os.path.exists(cfg):
        return ""
    print("ERROR: NGROK_AUTHTOKEN belum diisi.", file=sys.stderr)
    print("Dapatkan di https://dashboard.ngrok.com/get-started/your-authtoken", file=sys.stderr)
    print('Lalu: export NGROK_AUTHTOKEN="..."  atau  uv run python run_public.py --authtoken ...', file=sys.stderr)
    sys.exit(2)


def _normalize_domain(domain: str | None) -> str:
    if not domain:
        return ""
    d = domain.strip().replace("https://", "").replace("http://", "").rstrip("/").split("/")[0]
    return d


def start_ngrok(port: int, authtoken: str, domain: str | None = None) -> str:
    """Buka tunnel ngrok. Prioritas: binary sistem `ngrok` (mendukung --url reservasi),
    fallback ke pyngrok."""
    import shutil
    import subprocess

    domain = _normalize_domain(domain)
    ngrok_bin = shutil.which("ngrok")

    if ngrok_bin and domain:
        # pakai binary sistem + domain reservasi
        subprocess.run([ngrok_bin, "kill", "all"], capture_output=True)
        time.sleep(1)
        cmd = [ngrok_bin, "http", str(port), "--url", f"https://{domain}", "--log", "stdout"]
        if authtoken:
            cmd += ["--authtoken", authtoken]
        logf = open("/tmp/ngrok-rag.log", "a")
        subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        # tunggu tunnel online (polling)
        import urllib.request

        public_url = f"https://{domain}"
        for _ in range(30):
            time.sleep(2)
            try:
                req = urllib.request.Request(public_url, headers={"ngrok-skip-browser-warning": "true"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    if r.status in (200, 404):
                        # 404 = tunnel online tapi path / belum ada route (wajar, MCP di /mcp)
                        # cek bukan halaman error ngrok offline
                        body = r.read(2000).decode("utf-8", "ignore")
                        if "ERR_NGROK_3200" not in body:
                            return public_url
            except Exception:
                pass
        # kalau polling gagal, tetap return URL (mungkin MCP server belum jalan)
        return public_url

    # --- fallback pyngrok ---
    from pyngrok import conf, ngrok

    if authtoken:
        conf.get_default().auth_token = authtoken
    try:
        ngrok.kill()
    except Exception:
        pass
    time.sleep(1)
    if domain:
        # domain reservasi ngrok, contoh: alive-set-crappie.ngrok-free.app
        # pyngrok v3 -> pakai hostname; fallback ke url jika ditolak
        domain = domain.replace("https://", "").replace("http://", "").rstrip("/").split("/")[0]
        try:
            tunnel = ngrok.connect(port, "http", hostname=domain)
        except Exception:
            tunnel = ngrok.connect(port, "http", url=f"https://{domain}")
    else:
        tunnel = ngrok.connect(port, "http")
    public_url = tunnel.public_url
    assert public_url
    # pyngrok kadang return http://, paksa https untuk MCP client
    if public_url.startswith("http://"):
        public_url = public_url.replace("http://", "https://", 1)
    return public_url


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP RAG + ngrok public share")
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")))
    parser.add_argument("--authtoken", default=os.getenv("NGROK_AUTHTOKEN", ""))
    parser.add_argument(
        "--domain",
        default=os.getenv("NGROK_DOMAIN", "alive-set-crappie.ngrok-free.app"),
        help="domain reservasi ngrok (default dari env NGROK_DOMAIN)",
    )
    parser.add_argument("--no-prebuild", action="store_true")
    args = parser.parse_args()

    token = _get_token(args.authtoken)

    print(f"[ngrok] membuka tunnel ke localhost:{args.port} ...", flush=True)
    try:
        public_url = start_ngrok(args.port, token, args.domain)
    except Exception as e:
        print(f"[ngrok] GAGAL: {e}", file=sys.stderr)
        print("Pastikan authtoken benar dan koneksi internet tersedia.", file=sys.stderr)
        sys.exit(1)

    mcp_url = public_url.rstrip("/") + "/mcp"
    print("", flush=True)
    print("=" * 60, flush=True)
    print("MCP RAG PUBLIK AKTIF", flush=True)
    print(f"  Tunnel : {public_url} -> localhost:{args.port}", flush=True)
    print(f"  Endpoint MCP : {mcp_url}", flush=True)
    print("", flush=True)
    print("  Config untuk orang lain (opencode mcp.json):", flush=True)
    print(f'    {{"type": "streamable", "url": "{mcp_url}"}}', flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    # Jalankan MCP server di thread utama (blocking).
    # Set env agar mcp_server bind ke 0.0.0.0 dan build index.
    os.environ["MCP_PUBLIC"] = "1"
    os.environ["MCP_PORT"] = str(args.port)
    sys.argv = ["mcp_server", "--http", "--port", str(args.port)]
    if args.no_prebuild:
        sys.argv.append("--no-prebuild")

    # Import setelah env di-set agar rag_store baca config terbaru
    from retrieval_augmented_generation.mcp_server import main as mcp_main

    def _heartbeat() -> None:
        while True:
            time.sleep(300)
            print(f"[ngrok] tunnel masih aktif: {mcp_url}", flush=True)

    threading.Thread(target=_heartbeat, daemon=True).start()
    mcp_main()


if __name__ == "__main__":
    main()
