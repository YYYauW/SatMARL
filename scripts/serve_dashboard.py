from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    handler = lambda *items, **kwargs: SimpleHTTPRequestHandler(
        *items, directory=str(root), **kwargs
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"OASIS dashboard: http://127.0.0.1:{args.port}/web/oasis.html", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
