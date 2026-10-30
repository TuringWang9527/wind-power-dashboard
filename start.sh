#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ -x ".venv/bin/python" ]; then
    WIND_PYTHON=".venv/bin/python"
else
    WIND_PYTHON="python3"
fi
exec "$WIND_PYTHON" -m streamlit run app.py "$@"
