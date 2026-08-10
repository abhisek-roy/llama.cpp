# Launch Command Analysis

Traceability source commit: `b89f44654161`

This page analyzes the Linux server command currently used for Qwen3.6 27B Q8_0 over RPC.

```sh
/mnt/D/linux/Sources/llama.cpp/build/bin/llama-server \
  --model /mnt/F/LLMs/GGUFs/Qwen3.6/Qwen3.6-27B-Q8_0.gguf \
  --alias "Qwen3.6-27B" \
  -c 100000 \
  --fit on \
  --fit-target 128,128,512 \
  -t 10 \
  -tb 10 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00 \
  --host 0.0.0.0 \
  --port 7000 \
  --rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 \
  --parallel 4 \
  --kv-unified \
  --jinja \
  --no-mmap \
  --mmproj /mnt/D/LLMs/Qwen3.6/mmproj-Qwen3.6-27B-Q8_0.gguf \
  --image-min-tokens 1024 \
  --mmproj-offload \
  --reasoning-preserve \
  -lv 4 \
  -cram 32768
```

## What This Command Means

High-impact options:

| Option | Current value | Effect |
|---|---:|---|
| `--rpc` | `192.168.0.118:50052` | Registers the MacBook RPC device before `--device` is parsed. Keep this order. |
| `--device` | `CUDA0,CUDA1,RPC0` | Explicit device order. Layer split and `--fit-target` values follow this order. |
| `--fit` | `on` | Allows llama.cpp to adjust unset placement arguments so the model fits device memory. |
| `--fit-target` | `128,128,512` | Leaves 128 MiB on `CUDA0`, 128 MiB on `CUDA1`, and 512 MiB on `RPC0`. |
| `--tensor-split` | not set | Placement is automatic. With fit enabled, llama.cpp can set a split based on memory. |
| `-c` | `100000` | Large context. This reserves a large KV/cache budget and can force more work onto RPC. |
| `--parallel` | `4` | Four server slots. Good for concurrency, not always best for single-user latency. |
| `--kv-unified` | enabled | Uses one unified KV buffer shared across sequences. |
| `--mmproj` | set | Loads the multimodal projector. Useful for images, but consumes memory even in a text-heavy server profile. |
| `--no-mmap` | enabled | Affects loading and paging behavior more than steady-state token speed. |
| `-cram` | `32768` | Enables a 32 GiB prompt RAM cache. This helps repeated prompts, not raw token generation. |

## Main Placement Consequence

The current device order is:

```text
CUDA0 -> CUDA1 -> RPC0
4070 Ti Super -> 2060 Super -> MacBook RPC
```

In layer split mode, devices own contiguous layer ranges in device order. The output layer is treated as an extra offloadable layer and tends to live with the last layer segment.

That makes the current command risky for token generation:

- `RPC0` is the last device.
- The final layers and output layer can land on the MacBook.
- Logits or output tensors may need to come back over LAN on every generated token.
- With `--fit on` and no manual split, the fit path fills dense layers back-to-front when it must choose placement. In this order, the MacBook is considered first for the highest layers.

Prompt processing can still look reasonable because it works on many tokens at once. Token generation pays the synchronization and transfer cost every token, so this ordering can explain a profile like 150 tokens/s prompt processing and 12 tokens/s token generation.

## Baseline Placement Evidence

The baseline startup log confirmed:

```text
CUDA0 = NVIDIA GeForce RTX 4070 Ti SUPER, 15692 MiB free
CUDA1 = NVIDIA GeForce RTX 2060 SUPER, 7681 MiB free
RPC0  = MacBook M4 Pro via 192.168.0.118:50052, 18185 MiB free
```

`--fit on` did not change the initial placement:

```text
targets for free memory can be met on all devices, no changes needed
```

The model is fully offloaded:

```text
offloading output layer to GPU
offloading 64 repeating layers to GPU
offloaded 66/66 layers to GPU
```

Model buffers:

```text
CUDA0 model buffer = 9645.00 MiB
CUDA1 model buffer = 5016.75 MiB
RPC0  model buffer = 11310.53 MiB
Host  model buffer = 1288.28 MiB
```

KV buffers:

```text
CUDA0 KV buffer = 2346.00 MiB
CUDA1 KV buffer = 1173.00 MiB
RPC0  KV buffer = 2737.00 MiB
total KV        = 6256.00 MiB, 100096 cells, 16 layers
```

The KV split is 6/3/7 attention layers across CUDA0/CUDA1/RPC0. Because the model has `full_attention_interval = 4`, this maps to an approximate 24/12/28 repeating-layer split. This is a derived placement from the log, not an explicit per-layer dump.

Other confirmed settings:

```text
n_seq_max  = 4
n_ctx      = 100096
n_ctx_seq  = 100096
kv_unified = true
graph splits = 4
mmproj worst-case memory = 873.10 MiB
CLIP using CUDA0 backend
```

The important conclusion is that RPC0 currently has the largest model share and is the last device in the layer order. That puts the latency-sensitive end of the model on the MacBook RPC backend.

## Reordered Placement Evidence

The first reorder test used:

```text
--device RPC0,CUDA1,CUDA0
--fit-target 512,128,128
```

The startup log confirmed that llama.cpp used that order:

```text
using device RPC0  - MacBook RPC
using device CUDA1 - RTX 2060 SUPER
using device CUDA0 - RTX 4070 Ti SUPER
```

`--fit on` again made no placement change:

```text
targets for free memory can be met on all devices, no changes needed
```

Model buffers after reorder:

```text
RPC0  model buffer = 11187.75 MiB
CUDA1 model buffer = 5016.75 MiB
CUDA0 model buffer = 9767.78 MiB
Host  model buffer = 1288.28 MiB
```

Compared with the baseline, about 123 MiB moved from `RPC0` to `CUDA0`. This is consistent with the output layer moving from the RPC end of the stack to the 4070 Ti Super end of the stack.

KV buffers after reorder:

```text
RPC0  KV buffer = 2737.00 MiB
CUDA1 KV buffer = 1173.00 MiB
CUDA0 KV buffer = 2346.00 MiB
```

The KV sizes by device are unchanged, but their layer order changed. The likely attention-layer order is now:

```text
RPC0 first 7 attention layers -> CUDA1 next 3 attention layers -> CUDA0 final 6 attention layers
```

With `full_attention_interval = 4`, this is approximately:

```text
RPC0 first 28 repeating layers -> CUDA1 next 12 repeating layers -> CUDA0 final 24 repeating layers
```

The output buffer also changed from `CPU output buffer` in the baseline log to `CUDA_Host output buffer` in the reordered log.

Runtime result from this reorder test:

```text
prompt processing = about 179 tokens/s
token generation  = about 8.75 tokens/s
```

The prompt-processing number is not a useful improvement signal because it appears similar to the earlier run. Token generation got worse than the previously observed value around 12 tokens/s. This disproves the simple hypothesis that moving the output/final layers to the 4070 Ti Super is enough by itself. The next tests should reduce the RPC share while keeping all other variables fixed.

## Manual Split Evidence

The first reduced-RPC test used the original order:

```text
--device CUDA0,CUDA1,RPC0
--tensor-split 8,3,5
```

It did not fit at `-c 100000`:

```text
cudaMalloc failed: out of memory
failed to allocate CUDA0 buffer for kv cache
```

The next test used:

```text
--device CUDA0,CUDA1,RPC0
--tensor-split 7,3,6
```

Runtime result:

```text
prompt processing started around 240 tokens/s
prompt processing settled around 180 tokens/s by about 60K tokens
prompt processing dropped around 134 tokens/s while RPC cache writes were visible
token generation started around 8.6 tokens/s
token generation settled around 8.1-8.3 tokens/s
```

This split appears useful for long prompt ingestion but not for interactive token generation. If most sessions start from zero with long prompts, the prompt-processing gain may matter more than the lower token-generation rate. For short prompts or long generated answers, prefer the split with better token-generation speed.

The RPC server was running with local cache enabled, so cold runs can include extra `set_tensor` cache writes. For clean comparison, run each candidate twice with the same model and prompt, and compare the second run after the cache is warm.

## Impact-Ordered Speed Changes

### 1. Put RPC First And The Fastest Local GPU Last

Likely impact: high.

The first test should change placement order before changing many other flags. Put the remote device at the beginning of the layer stack and keep the fastest local GPU last so the final layers and output layer stay local.

Confirmed device order for this machine:

```sh
--rpc 192.168.0.118:50052 --device RPC0,CUDA1,CUDA0 --fit-target 512,128,128
```

The `--fit-target` values must move with the devices. Your current `512` MiB margin appears intended for the MacBook, so after moving `RPC0` to the front, use `512,128,128`.

Expected result:

- one network boundary remains, but earlier in the model
- output/logit readback should come from a local CUDA device instead of RPC
- auto-fit should prefer filling the last local GPU, then the other local GPU, then RPC

Verify device names first:

```sh
/mnt/D/linux/Sources/llama.cpp/build/bin/llama-server --rpc 192.168.0.118:50052 --list-devices
```

### 2. Add A Manual `--tensor-split`

Likely impact: high.

Use `--tensor-split` to reduce the MacBook share. Values follow the active device order.

For the original order:

```text
CUDA0,CUDA1,RPC0
```

the baseline KV split was approximately:

```text
6,3,7
```

so start by moving work from `RPC0` to `CUDA0`:

```text
--tensor-split 8,3,5
--tensor-split 9,3,4
```

For the reordered order:

If the order is:

```text
RPC0,CUDA1,CUDA0
```

use the mirrored tests:

```text
--tensor-split 5,3,8
--tensor-split 4,3,9
```

Read these as proportions, not layer counts. The goal is not to minimize RPC to zero immediately. The goal is to find the smallest RPC share that still fits the model and context without spilling to CPU or failing allocation.

For controlled split experiments:

1. Keep the prompt, image use, and context fixed.
2. Change only `--device`, `--fit-target`, and `--tensor-split`.
3. Record the startup log showing layer placement per device.
4. Record prompt processing and token generation speed.
5. Stop when generation speed gets worse or the model fails to fit.

If adding `--tensor-split` causes fit warnings or confusing auto-adjustment, run one diagnostic pass with:

```sh
--fit off --n-gpu-layers all
```

That makes the chosen split more explicit. It can fail with OOM; that is useful information during tuning.

### 3. Keep A Text-Only Profile Without `--mmproj`

Likely impact: high if memory pressure is causing extra RPC layers, low otherwise.

If you are not using image input for a run, test a text-only server profile:

```sh
--no-mmproj
```

or remove:

```text
--mmproj ...
--image-min-tokens 1024
--mmproj-offload
```

The multimodal projector itself is not the main token-generation loop, but it consumes memory. If removing it lets more transformer layers or KV cache stay on the local GPUs, token generation can improve.

### 4. Use A Smaller Context For Normal Runs

Likely impact: high when memory is tight.

`-c 100000` is expensive. It reserves a large KV/cache budget. It can also force more layers or KV onto the MacBook.

Create two profiles:

```text
fast profile: -c 32768 or -c 65536
long-context profile: -c 100000
```

If the smaller context lets you shift layers from `RPC0` to local CUDA, it can improve token generation more than a network upgrade.

### 5. Test `--parallel 1` For Single-User Latency

Likely impact: medium.

`--parallel 4` is a server throughput setting. It is useful when you expect concurrent requests. For one active chat, test:

```sh
--parallel 1
```

Keep `--parallel 4` as a throughput profile if you need multiple simultaneous slots. Compare per-request token generation speed, not only aggregate server throughput.

### 6. Improve The Network Path

Likely impact: medium to high if the link is 1 GbE or has router contention.

RPC is on the inference path, not only the model-load path. A USB-C Ethernet adapter through a router is often 1 GbE unless every port and adapter supports more.

Test in this order:

1. Confirm negotiated link speed on both hosts.
2. Use a quiet dedicated switch instead of a busy router path.
3. Try 2.5 GbE, 5 GbE, or 10 GbE adapters if both hosts can use them.
4. Try a direct wired host-to-host link if practical.

Keep the same `--device` and `--tensor-split` while changing the network, or the result will be hard to read.

### 7. Enable RPC Server Cache

Likely impact: high for repeated load time, low for steady token speed.

On the MacBook RPC host:

```sh
ggml-rpc-server --cache --host <trusted-host> --port 50052
```

This avoids resending large weight tensors that are already cached remotely. It does not remove per-token RPC synchronization during inference.

### 8. Test Local CUDA Transfer Options

Likely impact: medium for local multi-GPU transfers, low to medium overall while RPC remains the slowest part.

For the two local NVIDIA GPUs, test:

```sh
GGML_CUDA_P2P=1
```

Only keep it if output is stable and speed improves. Some consumer GPU and motherboard combinations do not support reliable peer-to-peer transfers.

For prompt processing throughput with multi-GPU pipeline work, also test:

```sh
CUDA_SCALE_LAUNCH_QUEUES=4x
```

This is more likely to help prompt processing than single-token generation.

### 9. Treat Prompt Cache As A Reuse Tool

Likely impact: high for repeated prompts, low for new-token generation.

You already use:

```sh
-cram 32768
```

That enables a large prompt RAM cache. It helps when requests share or repeat prompt prefixes. It does not make the model compute one new token faster.

`--cache-reuse` is also prompt-reuse oriented. In the current server code it is disabled for multimodal prompts, so do not use it as a first speed lever while `--mmproj` is loaded.

### 10. Try Speculative Decoding Only After Placement Is Stable

Likely impact: unknown until measured.

The commented draft MTP flags may help token generation if the draft path is supported and has good acceptance. It also adds memory and tuning complexity. Do not mix it into the first RPC placement experiments.

## First Experiment Sequence

Run these as separate launches and record logs.

### A. Current Baseline

Use your current command. Save:

- device list
- layer placement
- memory per device
- prompt processing tokens/s
- token generation tokens/s

### B. Reordered Devices, No Manual Split

Use the same command except:

```sh
--rpc 192.168.0.118:50052 --device RPC0,CUDA1,CUDA0 --fit-target 512,128,128
```

### C. Reordered Devices With Manual Split

Start with:

```sh
--rpc 192.168.0.118:50052 --device RPC0,CUDA1,CUDA0 --fit-target 512,128,128 --tensor-split 6,3,7
```

Then test:

```text
--tensor-split 5,3,8
--tensor-split 4,3,9
```

### D. Text-Only Fast Profile

Use the best placement from C, then remove multimodal loading:

```sh
--no-mmproj
```

### E. Smaller Context Fast Profile

Use the best placement from C or D, then test:

```text
-c 65536
-c 32768
```

### F. Single-User Latency Profile

Use the best placement from the earlier tests, then test:

```sh
--parallel 1
```

## What To Watch In Logs

The useful startup lines are the ones that show:

- available devices and free memory
- chosen split mode
- `n_gpu_layers`
- layer or tensor placement per backend
- KV cache size and backend placement
- `n_ctx`, `n_ctx_seq`, `n_seq_max`, and `kv_unified`
- whether pipeline parallelism is enabled or disabled

With `GGML_RPC_TELEMETRY=1`, also watch `rpc_telemetry:` lines:

- `sched`: backend split count, copied bytes, copy time, and compute time
- `graph client`: full graph send vs graph recompute and RPC command time
- `graph server`: remote backend compute time
- `cache`: hash hit or miss, skipped scope, and cache write intent
- `summary`: aggregate totals printed at normal process exit

For RPC specifically, pipeline parallelism should be expected to stay disabled because the RPC device reports no async compute or events. Therefore the highest-value tuning is placement: fewer RPC layers, RPC at the start of the stack, and output on local CUDA.
