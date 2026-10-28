#!/usr/bin/env bash
# Submit the E-STEER proof-of-concept as a DSMLP batch job.
#
# Run this FROM THE DSMLP LOGIN NODE (after `ssh <you>@dsmlp-login.ucsd.edu`),
# with dsmlp.env filled in next to this script (copy dsmlp.env.example first:
# `cp dsmlp.env.example dsmlp.env` and edit the FILL_IN values).
#
# What it does:
#   1. Fills in run_job.sh.template with your dsmlp.env values, producing a
#      concrete run_job.sh (git-ignored -- it embeds your own paths).
#   2. Submits it as a DSMLP batch job with launch-scipy-ml.sh -B (queues
#      the job and returns immediately; the job keeps running after you
#      disconnect). Inside the container, run_job.sh creates/reuses a venv
#      with --system-site-packages (so it sees the container's own
#      CUDA-matched PyTorch instead of reinstalling one from PyPI), pip
#      installs requirements-dsmlp.txt into it, then runs
#      `steer.py --device cuda` followed by `report.py`.
#
# Launch mechanics (launch-scipy-ml.sh, -B batch mode, -l gpu-class=...,
# resource flags) follow https://dsmlp-docs.ucsd.edu/running-jobs/launch-sh-reference/
# and https://dsmlp-docs.ucsd.edu/running-jobs/job-modes-and-limits/ --
# re-check those pages if DSMLP's documented flags change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/dsmlp.env"
TEMPLATE="${SCRIPT_DIR}/run_job.sh.template"
GENERATED="${SCRIPT_DIR}/run_job.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE}." >&2
    echo "Copy dsmlp.env.example to dsmlp.env and fill in the FILL_IN values first." >&2
    exit 1
fi
# shellcheck disable=SC1090
source "${ENV_FILE}"

for var in PROJECT_DIR_ON_DSMLP GPU_CLASS CPUS MEM_GB GPUS OUT_DIR VENV_DIR HF_HOME_DIR; do
    if [[ -z "${!var:-}" || "${!var}" == *FILL_IN* ]]; then
        echo "dsmlp.env: \$${var} is not set (still a FILL_IN placeholder?). Edit dsmlp.env and retry." >&2
        exit 1
    fi
done

sed \
    -e "s#__PROJECT_DIR_ON_DSMLP__#${PROJECT_DIR_ON_DSMLP}#g" \
    -e "s#__OUT_DIR__#${OUT_DIR}#g" \
    -e "s#__VENV_DIR__#${VENV_DIR}#g" \
    -e "s#__HF_HOME_DIR__#${HF_HOME_DIR}#g" \
    "${TEMPLATE}" > "${GENERATED}"
chmod +x "${GENERATED}"

echo "Submitting batch job:"
echo "  project dir : ${PROJECT_DIR_ON_DSMLP}"
echo "  gpu class   : ${GPU_CLASS}  (cpus=${CPUS} mem=${MEM_GB}G gpus=${GPUS})"
echo "  out dir     : ${OUT_DIR}"
echo "  venv        : ${VENV_DIR}"
echo "  generated   : ${GENERATED}  (seen in the container as ${PROJECT_DIR_ON_DSMLP}/dsmlp/run_job.sh)"

# Everything after `-B --` runs inside the container untouched, per
# https://dsmlp-docs.ucsd.edu/running-jobs/launch-sh-reference/
launch-scipy-ml.sh -c "${CPUS}" -m "${MEM_GB}" -g "${GPUS}" -l "gpu-class=${GPU_CLASS}" \
    -B -- bash "${PROJECT_DIR_ON_DSMLP}/dsmlp/run_job.sh"

echo
echo "Job submitted. It keeps running after you disconnect."
echo "Log file: ~/e-steer-dsmlp-<timestamp>.log (created once the container starts; tail -f it)."
echo "While the pod is alive you can also watch it per"
echo "https://dsmlp-docs.ucsd.edu/running-jobs/watching-your-job/ (kubectl get pods / kubectl logs)."
