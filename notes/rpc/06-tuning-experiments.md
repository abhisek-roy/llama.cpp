# Tuning Experiments

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

Use this page as a repeatable experiment log. Keep the model, prompt, context size, and build fixed when comparing runs.

## Baseline Information

Record:

- exact llama.cpp command
- model file and quantization
- source commit
- RPC server command
- network path, adapter model, router or switch, and negotiated link speed
- `--list-devices` output
- prompt processing tokens/s
- token generation tokens/s
- VRAM and memory use per device

## Experiment 1: Device Order

Goal: find the actual device order used by split controls.

```sh
llama-cli --rpc <mac-host>:50052 --list-devices
```

If you use explicit devices, write the exact order down. `--tensor-split` values follow that order.

## Experiment 2: Local-Only Baseline

Goal: know what the Linux host can do without the MacBook.

Example shape:

```sh
llama-cli -m <model.gguf> --device <local-gpu-0>,<local-gpu-1> --n-gpu-layers all
```

Compare:

- Does the model fit?
- If it fits, what are PP and TG speeds?
- If it does not fit, how many layers can be offloaded before failure?

## Experiment 3: Default RPC Split

Goal: measure the automatic free-memory split.

```sh
llama-cli -m <model.gguf> --rpc <mac-host>:50052 --n-gpu-layers all
```

Look for load logs that show devices and memory. If needed, use scheduler debug:

```sh
GGML_SCHED_DEBUG=1 llama-cli -m <model.gguf> --rpc <mac-host>:50052 --n-gpu-layers all
```

This can be noisy, but it helps show backend assignments and graph splits.

## Experiment 4: Bias Away From RPC

Goal: reduce the remote share while still fitting the model.

If device order is:

```text
RPC0, CUDA0, CUDA1
```

try a small matrix:

```text
--tensor-split 3,2,5
--tensor-split 2,2,6
--tensor-split 1,2,7
```

Stop when the model no longer fits or TG speed gets worse. The best split is likely not the one with maximum total memory use. It is the one that puts the latency-sensitive work on faster devices while keeping memory under control.

## Experiment 5: Network Path

Goal: determine whether the LAN path is limiting RPC.

Record the negotiated link speed for the Linux host, the MacBook USB-C Ethernet adapter, and the router or switch port. Then compare the normal router path with the fastest wired path available.

Test order:

1. Current USB-C Ethernet adapter through router.
2. Same adapter through a quiet dedicated switch, if available.
3. Faster 2.5 GbE, 5 GbE, or 10 GbE adapters, if available.
4. Direct wired host-to-host link, if practical.

Keep the same model command and `--tensor-split` while changing only the network path.

## Experiment 6: RPC Cache

Goal: reduce repeated model-load time without adding cache writes for transient inference tensors.

On the MacBook:

```sh
ggml-rpc-server --cache --host <trusted-host> --port 50052
```

Then run the same model twice. The second load should avoid sending already cached large weight tensors.

On the client, compare:

```text
GGML_RPC_CACHE_SCOPE=all
GGML_RPC_CACHE_SCOPE=weights
GGML_RPC_CACHE_SCOPE=none
```

Use `weights` as the first setting for normal testing with `--cache`. Use `none` as a diagnostic or when the server does not use `--cache`.

## Experiment 7: KV Offload Check

Goal: verify whether KV placement helps or hurts in your configuration.

Run once with default KV offload, then once with:

```sh
--no-kv-offload
```

Expected: for RPC layer placement, KV offload should usually be better because KV follows the layer device. If disabling it improves performance, that is a strong clue that remote KV traffic or backend scheduling needs deeper inspection.

## Experiment 8: RPC Telemetry

Goal: measure where RPC time is going before changing placement logic.

Run the RPC server and client with:

```text
GGML_RPC_TELEMETRY=1
```

Start with the useful current split:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 7,3,6
```

Run the same model and prompt with:

```text
GGML_RPC_CACHE_SCOPE=all
GGML_RPC_CACHE_SCOPE=weights
GGML_RPC_CACHE_SCOPE=none
```

In the logs, grep for:

```text
rpc_telemetry:
```

Compare:

- scheduler split `backend`, `copy_bytes`, `copy_ms`, and `compute_ms`
- RPC graph `mode=full` vs `mode=recompute`
- RPC graph client `serialize_ms` and `enqueue_ms`
- RPC graph server `compute_ms`
- cache `hash_hit`, `hash_miss`, `skipped_scope`, and `cache_write`

When the process exits normally, `GGML_RPC_TELEMETRY=1` also prints aggregate summary lines:

```text
rpc_telemetry: summary sched ...
rpc_telemetry: summary graph_client ...
rpc_telemetry: summary graph_server ...
rpc_telemetry: summary cache ...
```

If you want to summarize a saved log before shutdown, these bash commands are useful:

```sh
grep 'rpc_telemetry: sched ' run.log | awk '{
  delete kv;
  for (i = 1; i <= NF; i++) { split($i, a, "="); kv[a[1]] = a[2]; }
  b = kv["backend"];
  calls[b]++;
  bytes[b] += kv["copy_bytes"];
  copy[b] += kv["copy_ms"];
  compute[b] += kv["compute_ms"];
}
END {
  for (b in calls) {
    printf "sched backend=%s calls=%d copy_bytes=%d copy_ms=%.3f compute_ms=%.3f avg_copy_ms=%.3f avg_compute_ms=%.3f\n", b, calls[b], bytes[b], copy[b], compute[b], copy[b]/calls[b], compute[b]/calls[b];
  }
}'
```

```sh
grep 'rpc_telemetry: graph client ' run.log | awk '{
  delete kv;
  for (i = 1; i <= NF; i++) { split($i, a, "="); kv[a[1]] = a[2]; }
  k = kv["backend"] " " kv["mode"];
  calls[k]++;
  payload[k] += kv["payload_bytes"];
  enqueue[k] += kv["enqueue_ms"];
}
END {
  for (k in calls) {
    printf "graph_client %s calls=%d payload_bytes=%d enqueue_ms=%.3f avg_enqueue_ms=%.3f\n", k, calls[k], payload[k], enqueue[k], enqueue[k]/calls[k];
  }
}'
```

```sh
grep 'rpc_telemetry: cache client ' run.log | awk '{
  delete kv;
  for (i = 1; i <= NF; i++) { split($i, a, "="); kv[a[1]] = a[2]; }
  k = kv["event"] " " kv["usage"];
  calls[k]++;
  bytes[k] += kv["bytes"];
  cmd[k] += kv["cmd_ms"];
}
END {
  for (k in calls) {
    printf "cache_client %s calls=%d bytes=%d cmd_ms=%.3f\n", k, calls[k], bytes[k], cmd[k];
  }
}'
```

## Experiment 9: RPC Split Scale

Goal: test automatic placement with an RPC penalty while keeping the feature disabled by default.

Do not pass `--tensor-split` in this experiment. Start from the same device order:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0
```

Then compare:

```text
LLAMA_RPC_SPLIT_SCALE unset
LLAMA_RPC_SPLIT_SCALE=0.75
LLAMA_RPC_SPLIT_SCALE=0.50
LLAMA_RPC_SPLIT_SCALE=0.25
```

Use `GGML_RPC_TELEMETRY=1` and `GGML_RPC_CACHE_SCOPE=weights` during these runs. Compare the layer placement logs, PP/TG speed, and `rpc_telemetry:` split lines against the manual `--tensor-split 7,3,6` baseline.

## Experiment 10: Merged Async RPC Validation

Goal: validate upstream's queued RPC dispatcher and real event implementation against the pre-merge baseline.

There is no `GGML_RPC_PIPELINE` switch now. The RPC backend always reports async and event support. Graph compute, graph recompute, and backend async tensor operations are queued; events and synchronize calls wait on dispatcher fences.

Start from the current fitting launch:

```text
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
```

Compare the merged build against the recorded pre-merge run:

```text
pre-merge private fork with direct blocking RPC and optional fake events
post-merge fork with upstream dispatcher and real events
```

For measurement runs, also set:

```text
GGML_RPC_TELEMETRY=1
```

Watch for:

- startup line: `pipeline parallelism enabled`
- reservation fallback: `compute buffer allocation failed, retrying without pipeline parallelism`
- PP and TG speed
- `rpc_telemetry: summary sched ...` by backend
- graph-client `enqueue_ms`, not the legacy `cmd_ms`
- dependent `get_tensor ... wait_ms`
- changes in compute buffer memory pressure

Interpretation: low enqueue time only proves that work entered the dispatcher quickly. Compare server `compute_ms`, the next dependent read or event wait, and end-to-end PP/TG. A speedup requires useful independent work to overlap with the remote stage.

## Experiment 11: TCP Versus RDMA

Goal: measure whether the negotiated transport changes load time, boundary-copy time, or PP/TG.

RDMA is selected automatically when both peers support it. Confirm the startup logs, then repeat the same workload with TCP forced on both peers:

```text
GGML_RPC_NO_RDMA=1
```

Linux uses RoCE/InfiniBand through `libibverbs`. Supported Apple silicon systems can use RDMA over Thunderbolt through `librdma`; this requires macOS 26.2 or later and one-time `rdma_ctl enable` from Recovery. Connect using the address on the RDMA-capable link.

Keep model, context, device order, tensor split, cache warmth, and prompt fixed. Record the negotiated transport and compare model-load time, scheduler copy telemetry, PP, and TG.

## Experiment Log Template

```text
Date:
Source commit:
Model:
Command:
RPC server command:
Device order:
Tensor split:
Network path:
Negotiated link speed:
Context size:
Prompt tokens/s:
Generation tokens/s:
Telemetry summary:
Notes:
```
