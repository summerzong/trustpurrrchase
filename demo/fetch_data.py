#!/usr/bin/env python3
"""Download the challenge data files (git-ignored) into ../data/ using AIMO_API_KEY from .env.

  python3 fetch_data.py           # skips files that already exist
  python3 fetch_data.py --force   # re-download
"""
from __future__ import annotations

import os
import sys
import urllib.request

import agent
import catalog

BASE = os.environ.get("AIMO_API_BASE", "https://junction26.valioaimo.fi")
FILES = [  # (API download endpoint, local file the code expects)
    ("/api/products/download", catalog.PRODUCT_FILE),                   # ~87 MB
    ("/api/delivery-reliability/download", catalog.RELIABILITY_FILE),
    ("/api/carbon-footprint/download", catalog.CARBON_FILE),
]


def main():
    agent.load_dotenv()
    key = os.environ.get("AIMO_API_KEY")
    if not key:
        sys.exit("AIMO_API_KEY is missing: copy .env.example to .env and fill it in.")
    force = "--force" in sys.argv
    os.makedirs(catalog.DATA_DIR, exist_ok=True)
    for endpoint, path in FILES:
        name = os.path.basename(path)
        if os.path.exists(path) and not force:
            print(f"  exists   {name}")
            continue
        print(f"  download {name} ...", end="", flush=True)
        req = urllib.request.Request(BASE + endpoint, headers={
            "X-API-Key": key, "User-Agent": "trusted-trade-agent-demo/0.1"})  # Cloudflare blocks urllib's default UA
        tmp = path + ".part"
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:  # follows the 307 redirect
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, path)
        print(f" {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"Data ready in {os.path.normpath(catalog.DATA_DIR)}")


if __name__ == "__main__":
    main()
