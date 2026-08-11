# Async RPC Outcome Report

Private fork of llama.cpp. Scope: Linux host with `CUDA0` and `CUDA1`, plus MacBook M4 Pro as `RPC0` over wired 1 GbE. This report records the final state of the async investigation after telemetry runs. It replaces earlier speculative async notes that assumed RPC graph recompute blocked on the client.

## 1. Current useful setup

The current setup is worth keeping for long-context use:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
```

The investigation improved token generation from about 8 tokens/s to about 10-12 tokens/s, with prompt processing good enough for the target workflow. The MacBook RPC device is useful because it provides memory capacity for the model and long context. It is not a fast token-generation stage when it owns layers that every token must pass through.

The private fork should continue to track upstream llama.cpp changes. Keep the private controls opt-in and small so they remain easy to rebase.

## 2. Implemented private-fork controls

- `GGML_RPC_CACHE_SCOPE=all|weights|none`: controls whether the client uses the RPC cache protocol for all large tensors, only weights, or not at all. The useful setting here is `weights`.
- `GGML_RPC_TELEMETRY=1`: logs scheduler split timing, RPC graph client timing, RPC server graph timing, cache events, and `get_tensor` wait timing.
- `LLAMA_RPC_SPLIT_SCALE`: scales RPC reported free memory for automatic placement. Explicit `--tensor-split` is unchanged.
- `LLAMA_OUTPUT_LAYER_DEVICE=CUDA0`: places final norm and output head on the selected local CUDA device.
- `GGML_RPC_PIPELINE=1`: diagnostic only. It advertises async/events and uses client-side no-op RPC events so the scheduler can enter the pipeline path. It does not make RPC operations non-blocking.

## 3. What telemetry proved

The key measured sequence during token generation is:

```text
rpc_telemetry: graph client backend=RPC0[192.168.0.118:50052] mode=recompute ... cmd_ms=0.001
rpc_telemetry: sched split=3/5 backend=RPC0[192.168.0.118:50052] nodes=1369 ... compute_ms=0.005
rpc_telemetry: get_tensor client endpoint=192.168.0.118:50052 device=0 tensor=norm bytes=20480 wait_ms=52-53
rpc_telemetry: sched split=4/5 backend=CUDA0 nodes=3 ... copy_ms=52-53
```

Server telemetry on the MacBook showed:

```text
rpc_telemetry: graph server mode=recompute device=0 backend=MTL0 nodes=1369 compute_ms=about 50 status=0
```

Interpretation:

- `GRAPH_RECOMPUTE` is already fire-and-forget on the client. Its client `cmd_ms` is sub-millisecond.
- The MacBook Metal backend spends about 50 ms computing the RPC split.
- The Linux client waits about 52-53 ms in the next dependent `get_tensor`.
- The transferred tensor is only 20480 bytes, so this is not a bandwidth problem.
- The wait is dependency and server-completion time: CUDA0 needs RPC0's output before it can run the final norm/output split.

## 4. Why client-side async worker is not the next step

For the measured topology, the split order is effectively:

```text
CUDA0 layers -> CUDA1 layers -> RPC0 layers -> CUDA0 output
```

The CUDA0 output split needs the result from the RPC0 split. There is no independent CUDA split after RPC0 that can run while the MacBook computes. A client-side `GGML_RPC_ASYNC_WORKER=1` would make `get_tensor` return a future, but the very next split would immediately need that future's data. The wait would move from `get_tensor` to `event_wait`, `synchronize`, or a future wait. Token generation would not improve.

`GGML_RPC_PIPELINE=1` tested a related idea by allowing the scheduler pipeline path. It did not materially improve token generation and increased compute-buffer memory pressure. Keep it off for normal use.

## 5. What remains useful

The useful future work is operational tuning, not more async RPC machinery for this topology:

- Reduce RPC0's layer share if CUDA memory allows.
- Reduce KV/cache memory pressure if that lets more layers stay on local CUDA.
- Test lower context, lower KV cache types, or a smaller model when token generation speed matters more than maximum context.
- Treat `--split-mode row` or other tensor-parallel modes as a controlled experiment only. Over 1 GbE they are likely to increase synchronization frequency.
- Keep using `GGML_RPC_CACHE_SCOPE=weights` for warm weight-cache behavior without transient compute cache churn.
- Keep `LLAMA_OUTPUT_LAYER_DEVICE=CUDA0` when it fits and improves the output placement.

## 6. Stop point

This private fork has reached a practical stop point:

- PP is usable.
- TG improved from about 8 tokens/s to about 10-12 tokens/s.
- The setup fits the target long-context use case.
- The current private switches are useful for operation and diagnostics.
- Further async RPC work is deferred unless the placement or scheduler changes enough to create independent work that can overlap with RPC0.

Continue taking upstream llama.cpp changes and keep these notes as the baseline for future private-fork rebases.
