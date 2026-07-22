#!/usr/bin/env bash
# Backs up all three stateful stores for this project: ERPNext (bench), Qdrant
# (vector collection), and Langfuse Postgres. Wraps the individual commands
# documented in docs/DEPLOYMENT.md's "Backup & Restore" sections.
#
# Every path/name below is overridable via environment variable. None of the
# machine-specific ones (bench binary/dir, site name, backup destination) have
# a default baked in here -- put your machine's values in
# scripts/backup_all.local.sh (gitignored, sourced automatically if present),
# the same way CLAUDE.local.md holds machine-specific setup for this repo. See
# scripts/backup_all.local.sh.example for the format.
#
# Any step whose prerequisites aren't found (bench binary, docker container)
# is skipped with a warning rather than failing the whole run -- e.g. on a VPS
# with no local bench, or a partial stack.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/backup_all.local.sh" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/backup_all.local.sh"
fi

BENCH_BIN="${BENCH_BIN:-}"
BENCH_DIR="${BENCH_DIR:-}"
SITE_NAME="${SITE_NAME:-}"

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-contract}"
QDRANT_CONTAINER="${QDRANT_CONTAINER:-contract-intelligence-qdrant-1}"

POSTGRES_CONTAINER="${POSTGRES_CONTAINER:-contract-intelligence-postgres-1}"
POSTGRES_USER="${POSTGRES_USER:-langfuse}"
POSTGRES_DB="${POSTGRES_DB:-langfuse}"

BACKUP_ROOT="${BACKUP_ROOT:-$HOME/contract-intelligence-backups}"
QDRANT_BACKUP_DIR="$BACKUP_ROOT/qdrant"
POSTGRES_BACKUP_DIR="$BACKUP_ROOT/postgres"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$QDRANT_BACKUP_DIR" "$POSTGRES_BACKUP_DIR"

overall_status=0

echo "==> [1/3] ERPNext bench backup (${SITE_NAME:-<unset>})"
if [ -n "$BENCH_BIN" ] && [ -n "$BENCH_DIR" ] && [ -n "$SITE_NAME" ] && [ -x "$BENCH_BIN" ] && [ -d "$BENCH_DIR" ]; then
    if ( cd "$BENCH_DIR" && "$BENCH_BIN" --site "$SITE_NAME" backup --with-files ); then
        echo "    OK -> $BENCH_DIR/sites/$SITE_NAME/private/backups/"
    else
        echo "    FAILED (see bench output above)" >&2
        overall_status=1
    fi
else
    echo "    SKIPPED -- BENCH_BIN/BENCH_DIR/SITE_NAME not set or invalid." >&2
    echo "    Set them in scripts/backup_all.local.sh (see .example file)." >&2
fi
echo

echo "==> [2/3] Qdrant snapshot ($QDRANT_COLLECTION)"
if docker inspect "$QDRANT_CONTAINER" >/dev/null 2>&1; then
    snapshot_json="$(curl -sf -X POST "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots")"
    if [ -n "$snapshot_json" ]; then
        snapshot_name="$(echo "$snapshot_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])' 2>/dev/null)"
        if [ -n "$snapshot_name" ] \
            && docker cp "$QDRANT_CONTAINER:/qdrant/snapshots/$QDRANT_COLLECTION/$snapshot_name" "$QDRANT_BACKUP_DIR/" \
            && docker cp "$QDRANT_CONTAINER:/qdrant/snapshots/$QDRANT_COLLECTION/$snapshot_name.checksum" "$QDRANT_BACKUP_DIR/"; then
            local_sum="$(shasum -a 256 "$QDRANT_BACKUP_DIR/$snapshot_name" | awk '{print $1}')"
            expected_sum="$(cat "$QDRANT_BACKUP_DIR/$snapshot_name.checksum")"
            if [ "$local_sum" = "$expected_sum" ]; then
                echo "    OK, checksum verified -> $QDRANT_BACKUP_DIR/$snapshot_name"
            else
                echo "    CHECKSUM MISMATCH for $snapshot_name -- do not trust this copy" >&2
                overall_status=1
            fi
        else
            echo "    FAILED -- could not create/copy snapshot" >&2
            overall_status=1
        fi
    else
        echo "    FAILED -- snapshot API request returned nothing (is $QDRANT_URL reachable?)" >&2
        overall_status=1
    fi
else
    echo "    SKIPPED -- container $QDRANT_CONTAINER not running" >&2
fi
echo

echo "==> [3/3] Langfuse Postgres dump ($POSTGRES_DB)"
if docker inspect "$POSTGRES_CONTAINER" >/dev/null 2>&1; then
    dump_name="langfuse_${TIMESTAMP}.pgdump"
    if docker exec "$POSTGRES_CONTAINER" pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -F c -f "/tmp/$dump_name" \
        && docker cp "$POSTGRES_CONTAINER:/tmp/$dump_name" "$POSTGRES_BACKUP_DIR/" \
        && docker exec "$POSTGRES_CONTAINER" rm "/tmp/$dump_name"; then
        docker cp "$POSTGRES_BACKUP_DIR/$dump_name" "$POSTGRES_CONTAINER:/tmp/verify_$dump_name"
        if docker exec "$POSTGRES_CONTAINER" pg_restore --list "/tmp/verify_$dump_name" >/dev/null 2>&1; then
            echo "    OK, dump verified readable -> $POSTGRES_BACKUP_DIR/$dump_name"
        else
            echo "    DUMP UNREADABLE -- do not trust this copy" >&2
            overall_status=1
        fi
        docker exec "$POSTGRES_CONTAINER" rm "/tmp/verify_$dump_name" >/dev/null 2>&1
    else
        echo "    FAILED -- pg_dump or docker cp failed" >&2
        overall_status=1
    fi
else
    echo "    SKIPPED -- container $POSTGRES_CONTAINER not running" >&2
fi
echo

echo "Reminder: .env is not backed up by this script (secrets belong in a password"
echo "manager, not a file copy) -- see docs/DEPLOYMENT.md's .env section."

exit $overall_status
