"""JSON server on all IPv4 interfaces: python -m laya_local.server --port 8000."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import uuid
from pathlib import Path

from . import Client
from .api import BACKENDS, DEFAULT_BACKEND
from .snake import Game, step

MAX_BODY = 1024 * 1024


def handler_for(client):
    game = Game()
    revision = uuid.uuid4().hex

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/demo", "/snake"):
                body = Path(__file__).with_name("snake.html" if self.path == "/snake" else "demo.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/snake/state":
                self.reply(200, {"game": game.snapshot(), "revision": revision})
            elif self.path == "/health":
                self.reply(200, {"status": "ready", "backend": client.backend_name})
            elif self.path == "/models":
                availability=client.availability()
                max_len=max((int(s.rsplit("_l",1)[1]) for s in availability["supported_shapes"]),default=0)
                self.reply(200, {"models": [{"id": "english", "backend": client.backend_name,
                    "device": "npu", "question_types": ["choice", "score", "noul"],
                    "max_len": max_len, "head_max_len": 192, **availability}]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            nonlocal game, revision
            if self.path not in ("/predict", "/predict/batch", "/snake/reset", "/snake/step"):
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                if self.headers.get("Transfer-Encoding"):
                    raise ValueError("Use Content-Length, not Transfer-Encoding")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Request body must be between 1 byte and 1 MiB")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Incomplete request body")
                body = json.loads(raw)
                if self.path == "/snake/reset":
                    if not isinstance(body, dict) or set(body) - {"seed", "width", "height", "initial_length"}:
                        raise ValueError("Expected Snake seed, width, height, initial_length")
                    game = Game(**body)
                    revision = uuid.uuid4().hex
                    result = {"game": game.snapshot(), "revision": revision}
                elif self.path == "/snake/step":
                    if not isinstance(body, dict) or set(body) != {"revision", "shield"} or not isinstance(body["shield"], bool):
                        raise ValueError("Expected revision and boolean shield")
                    if body["revision"] != revision:
                        self.reply(409, {"error": "Game changed; reset or reload to continue"})
                        return
                    result = step(game, client, body["shield"])
                    revision = uuid.uuid4().hex
                    result["revision"] = revision
                else:
                    field = "state" if self.path == "/predict" else "states"
                    if not isinstance(body, dict) or set(body) != {field, "questions"}:
                        raise ValueError(f"Expected exactly {field} and questions")
                    method = client.predict if field == "state" else client.predict_batch
                    result = method(body[field], body["questions"])
            except (ValueError, UnicodeError) as exc:
                self.reply(400, {"error": str(exc)})
                return
            except Exception:
                logging.exception("Local inference failed")
                self.reply(500, {"error": "Inference failed; see server log"})
                return
            self.reply(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument('--backend', choices=BACKENDS, default=DEFAULT_BACKEND)
    parser.add_argument('--cache-entries',type=int,default=2)
    parser.add_argument('--cache-bytes',type=int,default=2*1024**3)
    parser.add_argument('--memo-entries',type=int,default=0)
    parser.add_argument('--memo-bytes',type=int,default=8*1024**2)
    parser.add_argument('--phase-major',action='store_true',help='Pair eligible independent B1/L64 batch requests')
    args = parser.parse_args()
    # A single server thread plus Client's lock keeps the shared runtime serial.
    with Client(backend=args.backend,cache_entries=args.cache_entries,cache_bytes=args.cache_bytes,
                memo_entries=args.memo_entries,memo_bytes=args.memo_bytes,phase_major=args.phase_major) as client:
        with HTTPServer(("0.0.0.0", args.port), handler_for(client)) as server:
            print(f"Laya listening on 0.0.0.0:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
