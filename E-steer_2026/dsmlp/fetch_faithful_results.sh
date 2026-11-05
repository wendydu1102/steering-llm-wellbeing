#!/usr/bin/env bash
# Run on your LAPTOP: copy a faithful run (results, figures, REPORT.md, logs) back from DSMLP.
# Usage: bash dsmlp/fetch_faithful_results.sh <run_id> [dest_dir]
# ssh-based copying does not need the VPN: https://dsmlp-docs.ucsd.edu/workspaces-and-storage/moving-and-sharing-data/
set -euo pipefail
RUN_ID="${1:?usage: fetch_faithful_results.sh <run_id> [dest_dir]}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "${SCRIPT_DIR}/dsmlp.env"
DEST="${2:-${SCRIPT_DIR}/../results_faithful}"
mkdir -p "${DEST}/${RUN_ID}"
HOST="${DSMLP_USERNAME}@dsmlp-login.ucsd.edu"
# Skip the bulky per-example generations unless you want them: add --exclude patterns as needed.
rsync -av "${HOST}:${LOGIN_NODE_PROJECT_DIR}/results_faithful/${RUN_ID}/" "${DEST}/${RUN_ID}/"
rsync -av "${HOST}:${LOGIN_NODE_PROJECT_DIR}/results_faithful/logs/e-steer-faithful-${RUN_ID}.log" "${DEST}/${RUN_ID}/container.log" || \
    echo "(container log not found yet)"
echo "Fetched to ${DEST}/${RUN_ID}  (open REPORT.md)"
