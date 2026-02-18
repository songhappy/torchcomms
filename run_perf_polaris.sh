CLUSTER=${CLUSTER:-polaris}
case "$CLUSTER" in
  borealis)
    CLUSTER_USER=guoqiong
    DEFAULT_TEST_BACKEND=xccl
    DEFAULT_TEST_DEVICE=xpu
    ;;
  polaris)
    CLUSTER_USER=songhappy
    DEFAULT_TEST_BACKEND=nccl
    DEFAULT_TEST_DEVICE=cuda
    ;;
  *)
    echo "Invalid CLUSTER=$CLUSTER. Use one of: borealis, polaris"
    exit 1
    ;;
esac

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
# AMD EPYC 7543P (1 socket, 4 NUMA nodes):
# node0: 0-7,32-39; node1: 8-15,40-47; node2: 16-23,48-55; node3: 24-31,56-63
# Default list is ordered to spread early ranks across NUMA nodes first.
CPU_BIND=${CPU_BIND:-list:0-3:8-11:16-19:24-27:4-7:12-15:20-23:28-31:32-35:40-43:48-51:56-59}
export CPU_BIND

case "$LOCAL_SIZE" in
  12)
    export CCL_WORKER_AFFINITY=2,10,18,26,6,14,22,30,34,42,50,58
    ;;
  8)
    export CCL_WORKER_AFFINITY=2,10,18,26,6,14,22,30
    ;;
  6)
    export CCL_WORKER_AFFINITY=2,10,18,26,6,14
    ;;
  4)
    export CCL_WORKER_AFFINITY=2,10,18,26
    ;;
  2)
    export CCL_WORKER_AFFINITY=2,10
    ;;
  1)
    export CCL_WORKER_AFFINITY=2
    ;;
  *)
    echo "Unsupported local size ${LOCAL_SIZE} for CPU binding" >&2
    exit 1
    ;;
esac

if [[ "$CPU_BIND" == list:* ]]; then
  IFS=":" read -ra __cpu_bind_entries <<< "${CPU_BIND#list:}"
  if (( ${#__cpu_bind_entries[@]} < LOCAL_SIZE )); then
    echo "CPU_BIND provides ${#__cpu_bind_entries[@]} entries but LOCAL_SIZE=${LOCAL_SIZE}" >&2
    exit 1
  fi
  unset __cpu_bind_entries
fi

COMM_IMPL=${COMM_IMPL:-both}

case "$COMM_IMPL" in
  both)
    impl_list="comms c10"
    ;;
  comms)
    impl_list="comms"
    ;;
  c10|c10d)
    impl_list="c10"
    ;;
  *)
    echo "Invalid COMM_IMPL=$COMM_IMPL. Use one of: both, comms, c10d"
    exit 1
    ;;
esac

site=${BASE_HOME}/miniforge3/envs/comms/lib/python3.10/site-packages/torchcomms
repo=${BASE_HOME}/git/torchcomms/comms/torchcomms

ln -sf $site/_comms*.so $repo/
[ -f $site/libtorchcomms.so ] && ln -sf $site/libtorchcomms.so $repo/

RESULTS_DIR=${RESULTS_DIR:-${BASE_HOME}/git/torchcomms/perf_results_binding}
mkdir -p "$RESULTS_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)

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
