from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 only behind a firewall or trusted network.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = lambda *items, **kwargs: SimpleHTTPRequestHandler(
        *items, directory=str(root), **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(
        f"Fast-Slow dashboard: http://{args.host}:{args.port}/web/fast_slow.html",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
