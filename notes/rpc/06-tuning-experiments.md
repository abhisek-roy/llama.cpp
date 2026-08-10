# Tuning Experiments

Traceability source commit: `b89f44654161`

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
- RPC graph client `cmd_ms`
- RPC graph server `compute_ms`
- cache `hash_hit`, `hash_miss`, `skipped_scope`, and `cache_write`

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
