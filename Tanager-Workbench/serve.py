"""Run the Tanager scene explorer locally on Windows.

This development server serves the static app files and connects the
``/api/spectrum``, ``/api/roi``, ``/api/composite``, ``/api/coastal`` and
``/api/ghg`` URLs
to the spectral
extraction code. It avoids requiring Vercel's local Python runtime.
"""
# app  code
from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from api.composite import composite_response
from api.coastal import coastal_response
from api.ghg import ghg_response
from api.roi import roi_response
from api.spectrum import SpectrumError, spectrum_response


APP_ROOT = Path(__file__).resolve().parent


class TanagerHandler(SimpleHTTPRequestHandler):
    """Serve app files normally and handle spectral API requests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_ROOT), **kwargs)

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if urlparse(self.path).path not in {"/api/spectrum", "/api/roi", "/api/composite", "/api/coastal", "/api/ghg"}:
            self.send_error(404)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/ghg":
            try:
                self.send_json(200, ghg_response(parse_qs(urlparse(self.path).query)))
            except SpectrumError as exc:
                self.send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                self.send_json(500, {"error": f"unexpected GHG API error: {exc}"})
            return
        if path == "/api/coastal":
            try:
                self.send_json(200, coastal_response(parse_qs(urlparse(self.path).query)))
            except SpectrumError as exc:
                self.send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                self.send_json(500, {"error": f"unexpected coastal API error: {exc}"})
            return
        if path == "/api/composite":
            try:
                png, metadata = composite_response(parse_qs(urlparse(self.path).query))
                metadata_text = json.dumps(metadata, separators=(",", ":"))
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Expose-Headers", "X-Tanager-Composite")
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("X-Tanager-Composite", metadata_text)
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            except SpectrumError as exc:
                self.send_json(exc.status, {"error": exc.message})
            except Exception as exc:
                self.send_json(500, {"error": f"unexpected composite API error: {exc}"})
            return
        if path != "/api/spectrum":
            super().do_GET()
            return

        try:
            params = parse_qs(urlparse(self.path).query)
            self.send_json(200, spectrum_response(params))
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # Keep unexpected failures readable in the UI.
            self.send_json(500, {"error": f"unexpected spectrum API error: {exc}"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/roi":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise SpectrumError(400, "request body is missing or too large")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise SpectrumError(400, "request body must be a JSON object")
            self.send_json(200, roi_response(payload))
        except json.JSONDecodeError:
            self.send_json(400, {"error": "request body is not valid JSON"})
        except SpectrumError as exc:
            self.send_json(exc.status, {"error": exc.message})
        except Exception as exc:  # Keep unexpected failures readable in the UI.
            self.send_json(500, {"error": f"unexpected ROI API error: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tanager map locally")
    parser.add_argument("--port", type=int, default=3000, help="local port (default: 3000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), TanagerHandler)
    print(f"Tanager explorer ready at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Tanager explorer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
