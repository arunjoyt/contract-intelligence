#!/usr/bin/env bash
# Runs the opt-in Playwright Desk/Streamlit E2E suite (tests/e2e/) against live
# services (ERPNext, Qdrant, OpenAI, a running app + Streamlit frontend -- see
# docs/IMPLEMENTATION_PLAN.md Step 17) and saves a durable copy of the results,
# mirroring scripts/run_integration.sh.
#
# One-time setup: playwright install chromium
#
# Usage:
#   ./scripts/run_e2e.sh                                    # full suite
#   ./scripts/run_e2e.sh tests/e2e/test_webhook_config.py    # one file (any pytest args)
#
# Writes to test-results/ (gitignored):
#   e2e_<timestamp>.log    -- full pytest -v output
#   e2e_<timestamp>.html   -- self-contained, browsable pass/fail report
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/test-results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/e2e_${TIMESTAMP}.log"
HTML_FILE="$RESULTS_DIR/e2e_${TIMESTAMP}.html"

mkdir -p "$RESULTS_DIR"

echo "==> Running E2E suite (RUN_E2E=1) -- drives a real browser against live services, can take a few minutes"

# pytest-playwright's own --output option defaults to "test-results" and
# shutil.rmtree()s it at session start to clear old trace/screenshot
# artifacts -- that collided with this script's own report directory of the
# same name (silently deleting it mid-run, before the html report could be
# written). Redirected to a subdirectory so it can't touch our log/html files.
RUN_E2E=1 "$REPO_ROOT/.venv/bin/pytest" "$REPO_ROOT/tests/e2e/" \
    -v --output="$RESULTS_DIR/playwright-artifacts" \
    --html="$HTML_FILE" --self-contained-html "$@" 2>&1 | tee "$LOG_FILE"
status="${PIPESTATUS[0]}"

echo
echo "==> Log:  $LOG_FILE"
echo "==> HTML report: $HTML_FILE"

exit "$status"
