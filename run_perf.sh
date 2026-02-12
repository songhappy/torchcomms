CLUSTER=${CLUSTER:-borealis}
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
export TEST_BACKEND=${TEST_BACKEND:-$DEFAULT_TEST_BACKEND}
export TEST_DEVICE=${TEST_DEVICE:-$DEFAULT_TEST_DEVICE}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
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

RESULTS_DIR=${RESULTS_DIR:-${BASE_HOME}/git/torchcomms/perf_results}
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
    echo "Command: torchrun --nproc_per_node=${NPROC_PER_NODE} -m ${test_module} all --iters 100"
    echo
  } | tee "$results_file"

  PYTHONPATH=${BASE_HOME}/git/torchcomms/comms \
  torchrun --nproc_per_node=${NPROC_PER_NODE} -m ${test_module} \
    all --iters 100 | tee -a "$results_file"
done
