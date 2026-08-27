# Reproduce Qwen3.8-27B-FP8 decode performance on Radeon R9700

This guide reproduces the accepted single-request decode configuration on one
eight-GPU Radeon AI PRO R9700 machine. TP4 and TP8 use the same source cutoffs,
model, image, and benchmark protocol, but intentionally use different attention
backends.

## Validated source cutoffs

| Repository | Branch | Validated runtime cutoff |
|---|---|---|
| `tanpinsiang/vllm` | `perf/rdna4-aiter-unified-kv-layout-v1` | `764dbb371cbb82c22f1a6fac726b010c01cad0db` |
| `tanpinsiang/aiter` | `perf/rdna4-unified-bf16-lds-v1` | `f07170b53a178d1de20001b130bb40b44127be12` |

The vLLM branch tip may contain a later documentation-only commit adding this
guide. Commit `764dbb371` is the runtime-code cutoff. Do not include later
split-K, speculative-decoding, or DFlash experiments when comparing results.

The accepted image is:

```text
docker.io/rocm/vllm-dev:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.26.0
sha256:4fa5bc9c24d25ef7e0fe38b23fc5f2b27fc986177bd2f9e496997bcf5af03871
```

Clone and verify both source trees:

```bash
git clone --branch perf/rdna4-aiter-unified-kv-layout-v1 \
  https://github.com/tanpinsiang/vllm.git vllm-rdna4
git clone --branch perf/rdna4-unified-bf16-lds-v1 \
  https://github.com/tanpinsiang/aiter.git aiter-rdna4

git -C vllm-rdna4 merge-base --is-ancestor \
  764dbb371cbb82c22f1a6fac726b010c01cad0db HEAD
test "$(git -C aiter-rdna4 rev-parse HEAD)" = \
  f07170b53a178d1de20001b130bb40b44127be12
```

## Prepare the pinned image overlay

The measured image predates the vLLM branch, while its compiled ROCm extensions
provide the validated ABI. The validation therefore overlaid the changed Python,
Triton, and configuration files on the image package and used AITER from the
checked-out source tree. The following constructs the same kind of overlay.

Set paths for the local machine. `HF_CACHE` must already contain the model if
offline mode is retained.

```bash
export VLLM_SRC="$PWD/vllm-rdna4"
export AITER_SRC="$PWD/aiter-rdna4"
export HF_CACHE=/path/to/huggingface-cache
export REPRO_ROOT="$PWD/qwen38-r9700-repro"
export IMAGE='docker.io/rocm/vllm-dev:rocm7.14.0_rdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.26.0@sha256:4fa5bc9c24d25ef7e0fe38b23fc5f2b27fc986177bd2f9e496997bcf5af03871'
export VLLM_SITE=/opt/python/lib/python3.14/site-packages/vllm

mkdir -p "$REPRO_ROOT/runtime" "$REPRO_ROOT/cache" "$REPRO_ROOT/results"
container_id=$(docker create "$IMAGE" /bin/true)
docker cp "$container_id:$VLLM_SITE" "$REPRO_ROOT/runtime/"
docker rm "$container_id"

git -C "$VLLM_SRC" diff --name-only \
  7ca49fbe4..764dbb371 -- vllm/ |
while IFS= read -r path; do
  install -D "$VLLM_SRC/$path" "$REPRO_ROOT/runtime/$path"
done
```

Use a new empty cache directory for each measured server. Reusing a warmed
cache is useful for normal serving but does not reproduce the fresh-server
validation protocol.

## Start the server

Choose one topology before running the common Docker command:

```bash
export TP=8  # Set to 4 or 8.
case "$TP" in
  4)
    export HIP_DEVICES=4,5,6,7
    export ATTN_BACKEND=ROCM_ATTN
    export AITER_UNIFIED_ATTN=0
    export FUSED_QK_CACHE=0
    export RDNA4_GDN_LAUNCH=0
    ;;
  8)
    export HIP_DEVICES=0,1,2,3,4,5,6,7
    export ATTN_BACKEND=ROCM_AITER_UNIFIED_ATTN
    export AITER_UNIFIED_ATTN=1
    export FUSED_QK_CACHE=1
    export RDNA4_GDN_LAUNCH=1
    ;;
  *)
    echo "TP must be 4 or 8" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# Give every fresh server its own empty compilation/JIT cache.
export RUN_CACHE="$REPRO_ROOT/cache/tp${TP}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_CACHE"
```

Start vLLM. The command deliberately enables only the AITER custom collective
and, on TP8, AITER unified attention. Other AITER subsystems stay disabled.

```bash
export CONTAINER="qwen38-r9700-tp${TP}"

docker run --detach --rm \
  --name "$CONTAINER" \
  --privileged \
  --network host \
  --ipc host \
  --ulimit nofile=1048576:1048576 \
  --mount "type=bind,src=$HF_CACHE,dst=/root/.cache/huggingface,readonly" \
  --mount "type=bind,src=$REPRO_ROOT/runtime/vllm,dst=$VLLM_SITE,readonly" \
  --mount "type=bind,src=$AITER_SRC,dst=/candidate-aiter,readonly" \
  --mount "type=bind,src=$RUN_CACHE,dst=/task-cache" \
  --mount "type=bind,src=$REPRO_ROOT/results,dst=/bench-results" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env VLLM_USAGE_STATS_SERVER=disabled \
  --env VLLM_CACHE_ROOT=/task-cache/vllm \
  --env TORCHINDUCTOR_CACHE_DIR=/task-cache/torchinductor \
  --env TRITON_CACHE_DIR=/task-cache/triton \
  --env AITER_JIT_DIR=/task-cache/aiter-jit \
  --env AITER_META_DIR=/candidate-aiter \
  --env AITER_REBUILD=1 \
  --env CK_DIR=/opt/python/lib/python3.14/site-packages/aiter_meta/3rdparty/composable_kernel \
  --env PYTHONPATH=/candidate-aiter \
  --env "HIP_VISIBLE_DEVICES=$HIP_DEVICES" \
  --env HSA_ENABLE_IPC_MODE_LEGACY=0 \
  --env HSA_NO_SCRATCH_RECLAIM=1 \
  --env HIP_FORCE_DEV_KERNARG=1 \
  --env SAFETENSORS_FAST_GPU=1 \
  --env VLLM_ROCM_USE_AITER=0 \
  --env VLLM_ROCM_USE_AITER_CUSTOM_AR=1 \
  --env VLLM_ROCM_USE_AITER_LINEAR=0 \
  --env VLLM_ROCM_USE_AITER_MOE=0 \
  --env VLLM_ROCM_USE_AITER_RMSNORM=0 \
  --env VLLM_ROCM_USE_AITER_MLA=0 \
  --env VLLM_ROCM_USE_AITER_MHA=0 \
  --env VLLM_ROCM_USE_AITER_FP8BMM=0 \
  --env VLLM_ROCM_USE_AITER_FP4BMM=0 \
  --env "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=$AITER_UNIFIED_ATTN" \
  --env VLLM_ROCM_USE_AITER_TRITON_GEMM=0 \
  --env VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
  --env VLLM_ROCM_USE_SKINNY_GEMM=1 \
  --env VLLM_GDN_DECODE_KERNEL=triton \
  --env VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
  --env "VLLM_ROCM_QWEN3_NEXT_FUSED_QK_CACHE=$FUSED_QK_CACHE" \
  --env "VLLM_ROCM_RDNA4_QWEN38_GDN_LAUNCH=$RDNA4_GDN_LAUNCH" \
  --entrypoint /opt/python/bin/vllm \
  "$IMAGE" serve Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size "$TP" \
  --language-model-only \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --kv-cache-dtype auto \
  --linear-backend triton \
  --attention-backend "$ATTN_BACKEND" \
  --disable-log-stats \
  --disable-uvicorn-access-log \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  -O2 \
  --compilation-config \
  '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"pass_config":{"fuse_allreduce_rms":true}}'
```

Wait for readiness and inspect the selected backends:

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 10; done
docker logs "$CONTAINER" 2>&1 | grep -E \
  'AITER_CUSTOM|ROCM_AITER_UNIFIED_ATTN|partitioned paged attention|local argmax|group-FP8'
```

TP4 should select `ROCM_ATTN`, partitioned hybrid-cache decode, and the exact
group-FP8 collective epilogue. TP8 should select
`ROCM_AITER_UNIFIED_ATTN`, the fused packed-KV QK writer, and vocabulary-parallel
local argmax. Both should show AITER custom all-reduce availability.

## Run the exact performance protocol

The first three requests are correctness/JIT warmup gates and are not reported.
The fourth command is the measured run: two 8,000-input/1,024-output requests,
served sequentially, with one additional benchmark warmup.

```bash
bench() {
  input=$1
  output=$2
  prompts=$3
  warmups=$4
  label=$5
  docker exec "$CONTAINER" /opt/python/bin/vllm bench serve \
    --model Qwen/Qwen3.8-27B-FP8 \
    --dataset-name random \
    --num-warmups "$warmups" \
    --random-input "$input" \
    --random-output "$output" \
    --request-rate 1 \
    --max-concurrency 1 \
    --temperature 0 \
    --num-prompts "$prompts" \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,99 \
    --save-result \
    --save-detailed \
    --result-dir /bench-results \
    --result-filename "tp${TP}-${label}.json"
}

bench 1024 16 1 0 gate-1024x16
bench 8192 16 1 0 gate-8192x16
bench 8000 1024 1 0 primary-setup
bench 8000 1024 2 1 primary
```

Stop the server after copying the result and logs:

```bash
docker logs "$CONTAINER" > "$REPRO_ROOT/results/tp${TP}-server.log" 2>&1
docker stop "$CONTAINER"
```

## Accepted results

Higher throughput and lower TPOT are better. Decode TPS below is
`1000 / mean TPOT`; vLLM's output-throughput field is lower because it includes
TTFT and the complete request duration.

| Topology | Mean TPOT | Decode TPS | Output throughput | Total throughput |
|---|---:|---:|---:|---:|
| TP4 | 16.633900 ms | 60.118 tok/s | 53.593 tok/s | 472.291 tok/s |
| TP8 | 11.514749 ms | 86.845 tok/s | 73.970 tok/s | 651.858 tok/s |

Both topology runs completed 2/2 measured requests and 2,048/2,048 requested
output tokens with zero measured-window JIT or fatal worker errors. The TP8
number is the mean of two fresh-server runs at 11.514705 and 11.514793 ms TPOT;
TP4 is the mean of two fresh-server runs at 16.634161 and 16.633638 ms TPOT.

The promotion accuracy gate used greedy decoding, 25-shot GSM8K, the first 100
questions, 256 maximum output tokens, and concurrency one. TP4 and TP8 both
scored 91/100 with zero invalid responses. This is a bounded regression check,
not a full-dataset accuracy claim.

Small timing differences are expected from GPU clocks, thermals, background
GPU users, model-cache placement, or a non-fresh compile cache. Do not compare a
run if another process owns any selected GPU or if compilation/autotuning occurs
during the measured window.
