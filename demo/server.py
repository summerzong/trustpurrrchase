#!/usr/bin/env python3
"""Minimal local web UI for the Trusted Trade Agent (standard library only).

  python3 server.py            ->  open http://127.0.0.1:8000 (classic UI) or http://127.0.0.1:8000/trustpurrrchase (TrustPurrrchase chat UI)
  HOST=0.0.0.0 HOME_UI=chat    ->  online deployment (Dockerfile): listen on every interface, "/" opens the chat UI
                                   and the classic UI moves to /classic (always available there)
  POST /api/request {text, use_llm, idempotency_key?, allow_duplicate?}  (or header Idempotency-Key)
  POST /api/decide {order_id, action, choose} · POST /api/relax {order_id, relaxation_id}
  POST /api/cancel {order_id} · POST /api/change {order_id, text, use_llm}   (before the order cut-off)
  POST /api/basket/preference {order_id, answer: price|delivery|stock|fit}
  POST /api/basket/confirm {order_id, choices: {line key: option id}, acks: [line key], quantities?: {line key: sales units}}
  request {simulate: "hospital_delay"} switches on a labelled fault-injection fixture (scenarios/)
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import agent
from aimo_api import AimoApiError

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML = os.path.join(HERE, "web", "index.html")
CHAT_HTML = os.path.join(HERE, "web", "trustpurrrchase.html")  # the TrustPurrrchase chat UI (Maito), wired to the same API
POLICY = os.path.join(HERE, "policy.json")
LOCK = threading.Lock()  # one request at a time: orders.json / prefs.json are plain files


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        home_chat = os.environ.get("HOME_UI") == "chat"
        path = self.path.split("?", 1)[0].split("#", 1)[0]  # hosts may append query parameters
        if path == "/classic" or (path in ("/", "/index.html") and not home_chat):
            with open(INDEX_HTML, "rb") as f:
                self._send(200, f.read(), "text/html")
        elif path in ("/trustpurrrchase", "/trustpurrrchase.html") or (home_chat and path in ("/", "/index.html")):
            with open(CHAT_HTML, "rb") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with LOCK:
                if self.path == "/api/request":
                    text = (body.get("text") or "").strip()
                    if not text:
                        raise ValueError("Please type a request.")
                    with open(POLICY, encoding="utf-8") as f:
                        policy = json.load(f)
                    result = agent.run_request(text, policy, use_llm=bool(body.get("use_llm", True)),
                                               idempotency_key=body.get("idempotency_key") or self.headers.get("Idempotency-Key"),
                                               allow_duplicate=bool(body.get("allow_duplicate")),
                                               simulate=body.get("simulate") or os.environ.get("AGENT_SIMULATE") or None)
                elif self.path == "/api/relax":
                    result = agent.relax_order(body["order_id"], body["relaxation_id"], body.get("by") or "buyer (web UI)")
                elif self.path == "/api/cancel":
                    result = agent.cancel_order(body["order_id"], body.get("by") or "buyer (web UI)")
                elif self.path == "/api/change":
                    text = (body.get("text") or "").strip()
                    if not text:
                        raise ValueError("Please type the changed request.")
                    with open(POLICY, encoding="utf-8") as f:
                        policy = json.load(f)
                    result = agent.change_order(body["order_id"], text, policy, body.get("by") or "buyer (web UI)",
                                                use_llm=bool(body.get("use_llm", True)))
                elif self.path == "/api/basket/preference":
                    result = agent.basket_preference(body["order_id"], body.get("answer"), body.get("by") or "buyer (web UI)")
                elif self.path == "/api/basket/confirm":
                    result = agent.basket_confirm(body["order_id"], body.get("choices") or {}, body.get("acks") or [],
                                                  body.get("by") or "buyer (web UI)", body.get("quantities") or {})
                elif self.path == "/api/decide":
                    result = agent.decide_order(body["order_id"], body.get("action") == "approve",
                                                int(body.get("choose") or 1), body.get("by") or "buyer (web UI)")
                else:
                    return self._send(404, {"error": "not found"})
            self._send(200, result)
        except (ValueError, KeyError) as e:
            self._send(400, {"error": str(e)})
        except AimoApiError as e:
            self._send(502, {"error": f"Valio Aimo API: {e}"})
        except Exception as e:  # keep the server alive, show the problem in the UI
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")


def main():
    agent.load_dotenv()
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")  # local only unless HOST is set (online deployment: 0.0.0.0)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Trusted Trade Agent UI -> http://{host}:{port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
