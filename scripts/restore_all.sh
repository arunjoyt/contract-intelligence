#!/usr/bin/env bash
# Restores all three stateful stores for this project: ERPNext (bench), Qdrant
# (vector collection), and Langfuse Postgres. Wraps the individual commands
# documented in docs/DEPLOYMENT.md's "Backup & Restore" sections.
#
# DESTRUCTIVE: each step overwrites live data (the ERPNext site's database and
# files, the Qdrant collection, the Langfuse database) with the backup you
# specify. Nothing is auto-selected -- every backup file path must be set
# explicitly in scripts/restore_all.local.sh (gitignored; see
# scripts/restore_all.local.sh.example for the format). A step whose file
# variable is unset/empty is skipped.
#
# Requires interactive confirmation before doing anything, unless --yes is
# passed (e.g. for scripted use).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/restore_all.local.sh" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/restore_all.local.sh"
else
    echo "scripts/restore_all.local.sh not found -- copy restore_all.local.sh.example" >&2
    echo "and point it at the exact backup files to restore. Aborting." >&2
    exit 1
fi

ASSUME_YES=0
if [ "${1:-}" = "--yes" ]; then
    ASSUME_YES=1
fi

BENCH_BIN="${BENCH_BIN:-}"
BENCH_DIR="${BENCH_DIR:-}"
SITE_NAME="${SITE_NAME:-}"
ERPNEXT_RESTORE_DB="${ERPNEXT_RESTORE_DB:-}"
ERPNEXT_RESTORE_PRIVATE_FILES="${ERPNEXT_RESTORE_PRIVATE_FILES:-}"
ERPNEXT_RESTORE_PUBLIC_FILES="${ERPNEXT_RESTORE_PUBLIC_FILES:-}"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-contract}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-contract-intelligence-qdrant-1}"
QDRANT_RESTORE_SNAPSHOT="${QDRANT_RESTORE_SNAPSHOT:-}"

POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-contract-intelligence-postgres-1}"
POSTGRES_USER="${POSTGRES_USER:-langfuse}"
POSTGRES_DB="${POSTGRES_DB:-langfuse}"
POSTGRES_RESTORE_DUMP="${POSTGRES_RESTORE_DUMP:-}"

echo "The following restores will run (each overwrites live data):"
echo
if [ -n "$ERPNEXT_RESTORE_DB" ]; then
    echo "  ERPNext ($SITE_NAME):"
    echo "    db:            $ERPNEXT_RESTORE_DB"
    echo "    private files: ${ERPNEXT_RESTORE_PRIVATE_FILES:-<none>}"
    echo "    public files:  ${ERPNEXT_RESTORE_PUBLIC_FILES:-<none>}"
else
    echo "  ERPNext: SKIPPED (ERPNEXT_RESTORE_DB not set)"
fi
if [ -n "$QDRANT_RESTORE_SNAPSHOT" ]; then
    echo "  Qdrant ($QDRANT_COLLECTION): $QDRANT_RESTORE_SNAPSHOT"
else
    echo "  Qdrant: SKIPPED (QDRANT_RESTORE_SNAPSHOT not set)"
fi
if [ -n "$POSTGRES_RESTORE_DUMP" ]; then
    echo "  Postgres ($POSTGRES_DB): $POSTGRES_RESTORE_DUMP"
else
    echo "  Postgres: SKIPPED (POSTGRES_RESTORE_DUMP not set)"
fi
echo

if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Type RESTORE to proceed: " confirm
    if [ "$confirm" != "RESTORE" ]; then
        echo "Aborted -- no changes made."
        exit 1
    fi
fi
echo

overall_status=0

echo "==> [1/3] ERPNext bench restore (${SITE_NAME:-<unset>})"
if [ -n "$ERPNEXT_RESTORE_DB" ]; then
    if [ -n "$BENCH_BIN" ] && [ -n "$BENCH_DIR" ] && [ -n "$SITE_NAME" ] \
        && [ -x "$BENCH_BIN" ] && [ -d "$BENCH_DIR" ] && [ -f "$ERPNEXT_RESTORE_DB" ]; then
        restore_args=(--site "$SITE_NAME" restore "$ERPNEXT_RESTORE_DB")
        if [ -n "$ERPNEXT_RESTORE_PRIVATE_FILES" ] && [ -f "$ERPNEXT_RESTORE_PRIVATE_FILES" ]; then
            restore_args+=(--with-private-files "$ERPNEXT_RESTORE_PRIVATE_FILES")
        fi
        if [ -n "$ERPNEXT_RESTORE_PUBLIC_FILES" ] && [ -f "$ERPNEXT_RESTORE_PUBLIC_FILES" ]; then
            restore_args+=(--with-public-files "$ERPNEXT_RESTORE_PUBLIC_FILES")
        fi
        if ( cd "$BENCH_DIR" && "$BENCH_BIN" "${restore_args[@]}" ); then
            echo "    OK"
        else
            echo "    FAILED (see bench output above)" >&2
            overall_status=1
        fi
    else
        echo "    SKIPPED -- BENCH_BIN/BENCH_DIR/SITE_NAME invalid or $ERPNEXT_RESTORE_DB missing" >&2
    fi
else
    echo "    SKIPPED -- ERPNEXT_RESTORE_DB not set"
fi
echo

echo "==> [2/3] Qdrant restore ($QDRANT_COLLECTION)"
if [ -n "$QDRANT_RESTORE_SNAPSHOT" ]; then
    if [ -f "$QDRANT_RESTORE_SNAPSHOT" ] && docker inspect "$QDRANT_CONTAINER" >/dev/null 2>&1; then
        checksum_file="${QDRANT_RESTORE_SNAPSHOT}.checksum"
        checksum_ok=1
        if [ -f "$checksum_file" ]; then
            local_sum="$(shasum -a 256 "$QDRANT_RESTORE_SNAPSHOT" | awk '{print $1}')"
            expected_sum="$(cat "$checksum_file")"
            if [ "$local_sum" != "$expected_sum" ]; then
                echo "    CHECKSUM MISMATCH -- refusing to restore a corrupt snapshot" >&2
                overall_status=1
                checksum_ok=0
            fi
        else
            echo "    WARNING -- no .checksum file next to snapshot, restoring unverified" >&2
        fi
        if [ "$checksum_ok" -eq 1 ]; then
            if curl -sf -X POST "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots/upload" \
                -F "snapshot=@$QDRANT_RESTORE_SNAPSHOT" >/dev/null; then
                echo "    OK -- restart the API so hybrid_search's BM25 index rebuilds from the restored data"
            else
                echo "    FAILED -- snapshot upload/restore request failed" >&2
                overall_status=1
            fi
        fi
    else
        echo "    SKIPPED -- $QDRANT_RESTORE_SNAPSHOT missing or container $QDRANT_CONTAINER not running" >&2
    fi
else
    echo "    SKIPPED -- QDRANT_RESTORE_SNAPSHOT not set"
fi
echo

echo "==> [3/3] Langfuse Postgres restore ($POSTGRES_DB)"
if [ -n "$POSTGRES_RESTORE_DUMP" ]; then
    if [ -f "$POSTGRES_RESTORE_DUMP" ] && docker inspect "$POSTGRES_CONTAINER" >/dev/null 2>&1; then
        dump_basename="$(basename "$POSTGRES_RESTORE_DUMP")"
        if docker cp "$POSTGRES_RESTORE_DUMP" "$POSTGRES_CONTAINER:/tmp/$dump_basename" \
            && docker exec "$POSTGRES_CONTAINER" pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
                --clean --if-exists "/tmp/$dump_basename"; then
            echo "    OK"
        else
            echo "    FAILED (see pg_restore output above)" >&2
            overall_status=1
        fi
        docker exec "$POSTGRES_CONTAINER" rm -f "/tmp/$dump_basename" >/dev/null 2>&1
    else
        echo "    SKIPPED -- $POSTGRES_RESTORE_DUMP missing or container $POSTGRES_CONTAINER not running" >&2
    fi
else
    echo "    SKIPPED -- POSTGRES_RESTORE_DUMP not set"
fi
echo

exit $overall_status
