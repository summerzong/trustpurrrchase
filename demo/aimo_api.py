"""Thin client for the Valio Aimo mock APIs (price + availability).

Every call is recorded (endpoint, time, product count, SHA-256 of the raw
response) so the audit trail can prove which data a decision was based on.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import aimo_contract

BASE = os.environ.get("AIMO_API_BASE", "https://junction26.valioaimo.fi")
MAX_PER_CALL = 1000


class AimoApiError(RuntimeError):
    pass


class AimoClient:
    def __init__(self, api_key=None, customer="HOTEL-DEMO-001"):
        self.api_key = api_key or os.environ.get("AIMO_API_KEY")
        if not self.api_key:
            raise AimoApiError("Set AIMO_API_KEY (the key from the Junction challenge page).")
        self.customer = customer
        self.calls = []  # evidence log for the audit trail

    def _post(self, path, body, retries=3):
        data = json.dumps(body).encode()
        for attempt in range(retries):
            req = urllib.request.Request(
                BASE + path, data=data, method="POST",
                headers={"X-API-Key": self.api_key, "Content-Type": "application/json",
                         "User-Agent": "trusted-trade-agent-demo/0.1"},  # Cloudflare blocks urllib's default UA
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < retries - 1:  # undisclosed rate limits
                    time.sleep(2 ** attempt)
                    continue
                raise AimoApiError(f"{path} -> HTTP {e.code}: {e.read()[:300]!r}")
            except urllib.error.URLError as e:
                if attempt < retries - 1:
                    time.sleep(1)
                    continue
                raise AimoApiError(f"{path} unreachable: {e.reason}")
        resp = json.loads(raw)
        # the contract check runs on the parsed response; the hash above is of the raw bytes received
        echo = aimo_contract.check_echo(resp, path, body["requestedDate"])
        self.calls.append({
            "endpoint": "POST " + path,
            "at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "requestedDate": body["requestedDate"],
            "products": len(body["products"]),
            "response_sha256": hashlib.sha256(raw).hexdigest(),
            **({"contract_anomalies": echo} if echo else {}),
        })
        if echo:
            resp["contract_anomalies"] = echo
        return resp

    def _batched(self, path, ids, date):
        merged = None
        for i in range(0, len(ids), MAX_PER_CALL):
            body = {"customer": self.customer, "requestedDate": date, "products": ids[i:i + MAX_PER_CALL]}
            resp = self._post(path, body)
            if merged is None:
                merged = resp
            else:
                merged["products"].extend(resp["products"])
                if resp.get("contract_anomalies"):
                    merged.setdefault("contract_anomalies", []).extend(resp["contract_anomalies"])
        return merged

    def price(self, ids, date):
        return self._batched("/api/price", ids, date)

    def availability(self, ids, date):
        return self._batched("/api/availability", ids, date)
