#!/bin/bash
set -euo pipefail
##############################################################################
# Unified performance-test launcher for torchcomms.
#
# Usage:
#   CLUSTER=borealis bash run_perf.sh          # XPU on Borealis (default)
#   CLUSTER=polaris  bash run_perf.sh          # CUDA on Polaris
#   NPROC_PER_NODE=2 COMM_IMPL=comms bash run_perf.sh
#
# Analyze results only (no benchmarks):
#   python3 comms/torchcomms/tests/perf/py/analyze_perf_report.py \
#       --perf-dir ~/git/torchcomms/perf_results_03242026
#
# Environment overrides (all optional):
#   CLUSTER          borealis | polaris              (default: borealis)
#   NPROC_PER_NODE   number of GPUs per node         (default: 4)
#   LOCAL_SIZE        defaults to NPROC_PER_NODE
#   COMM_IMPL        both | comms | c10d             (default: both)
#   TEST_BACKEND     xccl / nccl                     (auto from cluster)
#   TEST_DEVICE      xpu / cuda                      (auto from cluster)
#   CPU_BIND         taskset cpu-list spec            (auto from cluster)
#   RESULTS_DIR      where to write logs              (auto, date-stamped)
#   TORCHRUN_BIN     path to torchrun                 (auto-detected)
#   PYTHON_BIN       path to python                   (auto-detected)
##############################################################################

# ── Cluster configuration ──────────────────────────────────────────────────
CLUSTER=${CLUSTER:-borealis}
case "$CLUSTER" in
  borealis)
    CLUSTER_USER=guoqiong
    DEFAULT_TEST_BACKEND=xccl
    DEFAULT_TEST_DEVICE=xpu
    # Intel Data Center GPU Max (PVC), dual-socket Xeon topology
    DEFAULT_CPU_BIND="list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96"
    ;;
  polaris)
    CLUSTER_USER=songhappy
    DEFAULT_TEST_BACKEND=nccl
    DEFAULT_TEST_DEVICE=cuda
    # AMD EPYC 7543P (1 socket, 4 NUMA nodes)
    # node0: 0-7,32-39; node1: 8-15,40-47; node2: 16-23,48-55; node3: 24-31,56-63
    DEFAULT_CPU_BIND="list:0-3:8-11:16-19:24-27:4-7:12-15:20-23:28-31:32-35:40-43:48-51:56-59"
    ;;
  *)
    echo "Invalid CLUSTER=$CLUSTER. Use one of: borealis, polaris" >&2
    exit 1
    ;;
esac

# ── Environment setup ──────────────────────────────────────────────────────
BASE_HOME=/home/${CLUSTER_USER}
source "${BASE_HOME}/env-3.sh"

TORCHCOMMS_ENV_BIN="${BASE_HOME}/miniforge3/envs/comms/bin"
TORCHRUN_BIN=${TORCHRUN_BIN:-${TORCHCOMMS_ENV_BIN}/torchrun}
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_BIN=$(command -v torchrun)
fi
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found in PATH" >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-${TORCHCOMMS_ENV_BIN}/python}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=$(command -v python)
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "python not found in PATH" >&2
  exit 1
fi

export TORCHRUN_BIN
export PYTHON_BIN
export TEST_BACKEND=${TEST_BACKEND:-$DEFAULT_TEST_BACKEND}
export TEST_DEVICE=${TEST_DEVICE:-$DEFAULT_TEST_DEVICE}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
LOCAL_SIZE=${LOCAL_SIZE:-${NPROC_PER_NODE}}
export LOCAL_SIZE
CPU_BIND=${CPU_BIND:-$DEFAULT_CPU_BIND}
export CPU_BIND

# ── CCL worker affinity (per-cluster topology-aware) ───────────────────────
case "$CLUSTER" in
  borealis)
    case "$LOCAL_SIZE" in
      12) export CCL_WORKER_AFFINITY=5,13,21,29,37,45,57,65,73,81,89,97 ;;
      8)  export CCL_WORKER_AFFINITY=5,13,29,37,57,65,81,89 ;;
      6)  export CCL_WORKER_AFFINITY=5,21,37,57,73,89 ;;
      4)  export CCL_WORKER_AFFINITY=5,13,57,65 ;;
      2)  export CCL_WORKER_AFFINITY=5,57 ;;
      1)  export CCL_WORKER_AFFINITY=5 ;;
      *)  echo "Unsupported LOCAL_SIZE=${LOCAL_SIZE} for borealis CPU binding" >&2; exit 1 ;;
    esac
    ;;
  polaris)
    case "$LOCAL_SIZE" in
      12) export CCL_WORKER_AFFINITY=2,10,18,26,6,14,22,30,34,42,50,58 ;;
      8)  export CCL_WORKER_AFFINITY=2,10,18,26,6,14,22,30 ;;
      6)  export CCL_WORKER_AFFINITY=2,10,18,26,6,14 ;;
      4)  export CCL_WORKER_AFFINITY=2,10,18,26 ;;
      2)  export CCL_WORKER_AFFINITY=2,10 ;;
      1)  export CCL_WORKER_AFFINITY=2 ;;
      *)  echo "Unsupported LOCAL_SIZE=${LOCAL_SIZE} for polaris CPU binding" >&2; exit 1 ;;
    esac
    ;;
esac

# ── Validate CPU_BIND entry count ─────────────────────────────────────────
if [[ "$CPU_BIND" == list:* ]]; then
  IFS=":" read -ra __cpu_bind_entries <<< "${CPU_BIND#list:}"
  if (( ${#__cpu_bind_entries[@]} < LOCAL_SIZE )); then
    echo "CPU_BIND provides ${#__cpu_bind_entries[@]} entries but LOCAL_SIZE=${LOCAL_SIZE}" >&2
    exit 1
  fi
  unset __cpu_bind_entries
fi

# ── Comm implementation selection ──────────────────────────────────────────
COMM_IMPL=${COMM_IMPL:-both}
case "$COMM_IMPL" in
  both)    impl_list="comms c10" ;;
  comms)   impl_list="comms" ;;
  c10|c10d) impl_list="c10" ;;
  *)
    echo "Invalid COMM_IMPL=$COMM_IMPL. Use one of: both, comms, c10d" >&2
    exit 1
    ;;
esac

# ── CPU governor: performance mode (requires sudo) ────────────────────────
echo "Setting CPU governor to performance …"
# Uncomment the next line if you have passwordless sudo for cpupower:
# sudo /usr/bin/cpupower frequency-set -g performance
echo "Current CPU governors:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq
echo

# ── Symlink compiled extensions from site-packages into the repo tree ─────
site=${BASE_HOME}/miniforge3/envs/comms/lib/python3.10/site-packages/torchcomms
repo=${BASE_HOME}/git/torchcomms/comms/torchcomms

ln -sf $site/_comms*.so $repo/
[ -f $site/libtorchcomms.so ] && ln -sf $site/libtorchcomms.so $repo/
ln -sf $site/hooks/fr/_fr*.so $repo/hooks/fr/ 2>/dev/null || true

# ── Results directory (date-stamped) ───────────────────────────────────────
timestamp=$(date +%Y%m%d_%H%M%S)
datestamp=$(date +%m%d%Y)
RESULTS_DIR=${RESULTS_DIR:-${BASE_HOME}/git/torchcomms/perf_results_${datestamp}}
mkdir -p "$RESULTS_DIR"

# ── Run benchmarks ────────────────────────────────────────────────────────
for impl_tag in $impl_list; do
  if [ "$impl_tag" = "c10" ]; then
    test_module="torchcomms.tests.perf.py.collective_perf_test_c10d"
  else
    test_module="torchcomms.tests.perf.py.collective_perf_test"
  fi

  results_file="$RESULTS_DIR/collective_perf_${timestamp}_${TEST_DEVICE}_${impl_tag}.log"

  {
    echo "Collective perf run started: $(date -Iseconds)"
    echo "Host: $(hostname)"
    echo "Cluster: ${CLUSTER}"
    echo "Cluster User: ${CLUSTER_USER}"
    echo "Backend: ${TEST_BACKEND}"
    echo "Device: ${TEST_DEVICE}"
    echo "Comm Impl: ${impl_tag}"
    echo "nproc_per_node: ${NPROC_PER_NODE}"
    echo "CPU Bind: ${CPU_BIND}"
    echo "CCL Worker Affinity: ${CCL_WORKER_AFFINITY}"
    echo "Command: ${TORCHRUN_BIN} --nproc_per_node=${NPROC_PER_NODE} --no-python bash -c '<per-rank taskset launch>'"
    echo
  } | tee "$results_file"

  PYTHONPATH=${BASE_HOME}/git/torchcomms/comms \
  TEST_MODULE="${test_module}" \
  "${TORCHRUN_BIN}" --nproc_per_node="${NPROC_PER_NODE}" --no-python bash -c '
    if [[ -n "${CPU_BIND:-}" ]]; then
      if [[ "${CPU_BIND}" == list:* ]]; then
        IFS=":" read -ra __cpu_bind_entries <<< "${CPU_BIND#list:}"
        __rank_cpu="${__cpu_bind_entries[$LOCAL_RANK]}"
      else
        __rank_cpu="${CPU_BIND}"
      fi
    fi
    if [[ -n "${__rank_cpu:-}" ]]; then
      echo "[cpu_bind] rank=${LOCAL_RANK} cpus=${__rank_cpu}" >&2
      exec taskset -c "${__rank_cpu}" "${PYTHON_BIN}" -m "${TEST_MODULE}" all --iters 100
    fi
    exec "${PYTHON_BIN}" -m "${TEST_MODULE}" all --iters 100
  ' | tee -a "$results_file"
done

# ── Restore CPU governor ──────────────────────────────────────────────────
echo
echo "Restoring CPU governor to ondemand …"
# Uncomment the next line if you have passwordless sudo for cpupower:
# sudo /usr/bin/cpupower frequency-set -g ondemand
echo "Current CPU governors:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq

# ── Analyze results ───────────────────────────────────────────────────────
echo
echo "Running perf analysis …"
"${PYTHON_BIN}" "${BASE_HOME}/git/torchcomms/comms/torchcomms/tests/perf/py/analyze_perf_report.py" \
  --perf-dir "$RESULTS_DIR"
