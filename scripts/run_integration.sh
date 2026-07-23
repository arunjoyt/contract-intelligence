#!/usr/bin/env bash
# Runs the opt-in integration suite (tests/test_integration.py) against live
# services (ERPNext, Qdrant, OpenAI, and whatever else each test group needs --
# see the module docstring) and saves a durable copy of the results, since
# `pytest -v` output otherwise only lives in your terminal scrollback.
#
# Usage:
#   ./scripts/run_integration.sh                  # full suite
#   ./scripts/run_integration.sh -m langfuse       # one group (any pytest args)
#
# Writes to test-results/ (gitignored):
#   integration_<timestamp>.log   -- full pytest -v output
#   integration_<timestamp>.xml   -- JUnit XML, for tooling/CI dashboards
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/test-results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/integration_${TIMESTAMP}.log"
XML_FILE="$RESULTS_DIR/integration_${TIMESTAMP}.xml"

mkdir -p "$RESULTS_DIR"

echo "==> Running integration suite (RUN_INTEGRATION=1) -- this hits live services and can take several minutes"

RUN_INTEGRATION=1 "$REPO_ROOT/.venv/bin/pytest" "$REPO_ROOT/tests/test_integration.py" \
    -v --junitxml="$XML_FILE" "$@" 2>&1 | tee "$LOG_FILE"
status="${PIPESTATUS[0]}"

echo
echo "==> Log:  $LOG_FILE"
echo "==> JUnit XML: $XML_FILE"

exit "$status"
