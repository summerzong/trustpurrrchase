#!/bin/sh
# Online deployment entry point (used by the Dockerfile).
# Needs AIMO_API_KEY (and optionally GEMINI_API_KEY) as environment secrets.
# The challenge data is not in the repository: it is downloaded on first start.
set -e
cd "$(dirname "$0")"
if [ ! -f "${AIMO_DATA_DIR:-../data}/valio_aimo_junction26_product_data.json" ]; then
  echo "Downloading the Valio Aimo challenge data ..."
  python3 fetch_data.py
fi
export HOST="${HOST:-0.0.0.0}" HOME_UI="${HOME_UI:-chat}"
exec python3 server.py
