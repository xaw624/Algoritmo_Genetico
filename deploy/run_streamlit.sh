#!/usr/bin/env bash
# Lanzador usado por streamlit@.service
# Destino: /usr/local/bin/run_streamlit.sh (root:root, 755)
set -euo pipefail

# Variables esperadas desde /etc/streamlit/<instancia>.env:
# PYTHON_BIN, APP_FILE, PORT, EXTRA_ARGS (opcional)
# EXTRA_ARGS va sin comillas a propósito: puede contener varios argumentos.
# shellcheck disable=SC2086
exec "${PYTHON_BIN}" -m streamlit run "${APP_FILE}" \
  --server.port="${PORT}" \
  --server.address=127.0.0.1 \
  --server.headless=true \
  --browser.gatherUsageStats=false \
  ${EXTRA_ARGS:-}
