#!/usr/bin/env bash
# Run the team integration test suite against a real Ollama instance.
#
# Usage:
#   bash scripts/integration-test.sh [pytest-args...]
#
# Environment variables (all optional):
#   TEAM_INTEGRATION_OLLAMA_URL  — use an already-running Ollama instead of
#                                   starting a container (skips docker entirely)
#   TEAM_INTEGRATION_MODEL       — model to pull and use (default: smollm2:135m)
#   TEAM_INTEGRATION_PORT        — host port for the managed container (default: 11499)
#
# The script starts a temporary Ollama container, pulls the chosen model, runs
# pytest against tests/integration/, then tears the container down on exit.
# If TEAM_INTEGRATION_OLLAMA_URL is already set the container lifecycle is
# skipped and the existing instance is used as-is.

set -euo pipefail

MODEL="${TEAM_INTEGRATION_MODEL:-smollm2:135m}"
OLLAMA_PORT="${TEAM_INTEGRATION_PORT:-11499}"
CONTAINER_NAME="team-inttest-$$"   # PID-based name — unique per invocation

# ------------------------------------------------------------------ #
# Cleanup trap
# ------------------------------------------------------------------ #

_CONTAINER_STARTED=0

cleanup() {
    if [[ "$_CONTAINER_STARTED" -eq 1 ]]; then
        echo ""
        echo "→ Removing Ollama container ${CONTAINER_NAME} ..."
        docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------ #
# Ollama setup
# ------------------------------------------------------------------ #

if [[ -n "${TEAM_INTEGRATION_OLLAMA_URL:-}" ]]; then
    echo "→ Using external Ollama at ${TEAM_INTEGRATION_OLLAMA_URL}"
    echo "→ Skipping container lifecycle."
else
    echo "→ Starting Ollama container '${CONTAINER_NAME}' on port ${OLLAMA_PORT} ..."
    docker run -d \
        --name "${CONTAINER_NAME}" \
        -p "${OLLAMA_PORT}:11434" \
        ollama/ollama:latest \
        >/dev/null
    _CONTAINER_STARTED=1

    export TEAM_INTEGRATION_OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"

    echo "→ Waiting for Ollama to be ready ..."
    READY=0
    for _i in $(seq 1 60); do
        if curl -sf "${TEAM_INTEGRATION_OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
            READY=1
            break
        fi
        sleep 1
    done
    if [[ "$READY" -eq 0 ]]; then
        echo "✗ Ollama did not become ready within 60 s. Check docker logs ${CONTAINER_NAME}." >&2
        exit 1
    fi
    echo "✓ Ollama is ready."

    echo "→ Pulling model '${MODEL}' (this may take a few minutes on first run) ..."
    docker exec "${CONTAINER_NAME}" ollama pull "${MODEL}"
    echo "✓ Model '${MODEL}' is ready."
fi

export TEAM_INTEGRATION_MODEL="${MODEL}"

# ------------------------------------------------------------------ #
# Run tests
# ------------------------------------------------------------------ #

echo ""
echo "→ Running integration tests ..."
echo ""

python -m pytest tests/integration/ -v --tb=short "$@"
