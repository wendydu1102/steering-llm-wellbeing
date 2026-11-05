#!/usr/bin/env bash
# Submit the paper-faithful E-STEER pipeline (faithful/) as a DSMLP batch job.
# Run FROM THE DSMLP LOGIN NODE with dsmlp.env filled in (cp dsmlp.env.example dsmlp.env).
#
# Launch mechanics follow https://dsmlp-docs.ucsd.edu/running-jobs/launch-sh-reference/ and
# .../job-modes-and-limits/: launch-scipy-ml.sh -c <cpus> -m <GB> -g <n> -l gpu-class=<class> -B -- <command>,
# with the runtime limit raised via K8S_TIMEOUT_SECONDS (default 6 h, max settable 12 h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/dsmlp.env"
TEMPLATE="${SCRIPT_DIR}/run_faithful.sh.template"
GENERATED="${SCRIPT_DIR}/run_faithful.sh"

[[ -f "${ENV_FILE}" ]] || { echo "Missing ${ENV_FILE}: cp dsmlp.env.example dsmlp.env and fill it in." >&2; exit 1; }
# shellcheck disable=SC1090
source "${ENV_FILE}"

for var in PROJECT_DIR_ON_DSMLP FAITHFUL_GPU_CLASS FAITHFUL_CPUS FAITHFUL_MEM_GB FAITHFUL_RUNTIME_HOURS \
           FAITHFUL_PROFILE FAITHFUL_STAGES FAITHFUL_OUT_DIR FAITHFUL_VENV_DIR FAITHFUL_HF_HOME_DIR; do
    if [[ -z "${!var:-}" || "${!var}" == *FILL_IN* ]]; then
        echo "dsmlp.env: \$${var} is not set. Edit dsmlp.env and retry." >&2; exit 1
    fi
done
if [[ -z "${FAITHFUL_ACK:-}" ]]; then
    echo "dsmlp.env: FAITHFUL_ACK is empty. This run substitutes components the paper does not specify" >&2
    echo "(SAE checkpoint, judge model, alpha, subset sizes, possibly a smaller model, ...). Read faithful/README.md, then list the" >&2
    echo "ones you accept, e.g. FAITHFUL_ACK=sae-qwen-scope,judge-backbone,alpha,subset-sizes" >&2
    exit 1
fi
FAITHFUL_MODEL_TIER="${FAITHFUL_MODEL_TIER:-auto}"
case "${FAITHFUL_GPU_CLASS}" in
    extra-small) echo "ERROR: extra-small (~6 GB) is too small for both tiers (Qwen3-8B ~21 GB, Qwen3-1.7B ~7-11 GB)." >&2; exit 1 ;;
    small) echo "NOTE: small (~12 GB) cannot hold Qwen3-8B; tier '${FAITHFUL_MODEL_TIER}' will resolve to the smaller Qwen3-1.7B (needs ack smaller-model)." >&2 ;;
    medium|large|extra-large) ;;
    *) echo "Unknown GPU class '${FAITHFUL_GPU_CLASS}' (see https://dsmlp-docs.ucsd.edu/gpu-access/gpu-classes/)" >&2; exit 1 ;;
esac
case "${FAITHFUL_MODEL_TIER}" in auto|qwen3-8b|qwen3-1.7b) ;; *) echo "FAITHFUL_MODEL_TIER must be auto, qwen3-8b or qwen3-1.7b" >&2; exit 1 ;; esac
if (( FAITHFUL_RUNTIME_HOURS > 12 )); then
    echo "FAITHFUL_RUNTIME_HOURS=${FAITHFUL_RUNTIME_HOURS} exceeds the documented 12 h maximum settable at launch." >&2; exit 1
fi

RUN_ID="${FAITHFUL_RUN_ID:-${FAITHFUL_PROFILE}-$(date +%Y%m%d-%H%M%S)}"

sed \
    -e "s#__RUN_ID__#${RUN_ID}#g" \
    -e "s#__PROJECT_DIR__#${PROJECT_DIR_ON_DSMLP}#g" \
    -e "s#__OUT_DIR__#${FAITHFUL_OUT_DIR}#g" \
    -e "s#__VENV_DIR__#${FAITHFUL_VENV_DIR}#g" \
    -e "s#__HF_HOME_DIR__#${FAITHFUL_HF_HOME_DIR}#g" \
    -e "s#__ACK__#${FAITHFUL_ACK}#g" \
    -e "s#__NRC_VAD_PATH__#${FAITHFUL_NRC_VAD_PATH:-}#g" \
    -e "s#__MODEL_TIER__#${FAITHFUL_MODEL_TIER}#g" \
    -e "s#__PROFILE__#${FAITHFUL_PROFILE}#g" \
    -e "s#__STAGES__#${FAITHFUL_STAGES}#g" \
    "${TEMPLATE}" > "${GENERATED}"
chmod +x "${GENERATED}"

export K8S_TIMEOUT_SECONDS=$(( 3600 * FAITHFUL_RUNTIME_HOURS ))

echo "Submitting paper-faithful job:"
echo "  run id      : ${RUN_ID}  (resubmit with FAITHFUL_RUN_ID=${RUN_ID} to resume)"
echo "  profile     : ${FAITHFUL_PROFILE}   stages: ${FAITHFUL_STAGES}   model tier: ${FAITHFUL_MODEL_TIER}"
echo "  gpu class   : ${FAITHFUL_GPU_CLASS}  (cpus=${FAITHFUL_CPUS} mem=${FAITHFUL_MEM_GB}G, runtime<=${FAITHFUL_RUNTIME_HOURS}h)"
echo "  results     : ${FAITHFUL_OUT_DIR}/${RUN_ID}/"
echo "  container log: ${FAITHFUL_OUT_DIR}/logs/e-steer-faithful-${RUN_ID}.log"

launch-scipy-ml.sh -c "${FAITHFUL_CPUS}" -m "${FAITHFUL_MEM_GB}" -g 1 -l "gpu-class=${FAITHFUL_GPU_CLASS}" \
    -B -- bash "${PROJECT_DIR_ON_DSMLP}/dsmlp/run_faithful.sh"

echo
echo "Submitted. A GPU request can wait in Pending (OnDemandLeaseDenied) until capacity or budget is available:"
echo "https://dsmlp-docs.ucsd.edu/running-jobs/watching-your-job/"
echo "Retrieve results from your laptop with: bash dsmlp/fetch_faithful_results.sh ${RUN_ID}"
