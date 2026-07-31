from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve artifacts and make an unstarted pipeline explicit without 404 spam."""

    def do_GET(self) -> None:
        request_path = urlparse(self.path).path
        if request_path.startswith("/runs/") and request_path.endswith(
            "/pipeline.json"
        ):
            artifact = Path(self.directory) / request_path.lstrip("/")
            if not artifact.is_file():
                payload = json.dumps(
                    {
                        "status": "waiting",
                        "stage": "not_started",
                        "message": (
                            f"No pipeline.json at {request_path}. Start the pipeline "
                            "or select the correct run root."
                        ),
                        "training_runs": [],
                        "stages": [],
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        super().do_GET()


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
    handler = lambda *items, **kwargs: DashboardHandler(
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
