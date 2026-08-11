# RPC Performance Implementation Plan

Traceability source commit: `59abc3967`

This plan breaks [RPC performance system design](09-performance-system-design.md) into small private-fork changes. Do not submit these changes upstream without a separate discussion.

## Ground Rules

- Keep each change opt-in unless it is preserving existing behavior.
- Do not change placement behavior before telemetry exists.
- Do not commit or push from an agent session.
- The user runs CMake build commands.
- Prefer notes updates with each code change so the system model stays current.

## Final Investigation Status

This private-fork investigation is complete enough for continued use. The current setup gives good prompt processing for the target workflow and improved token generation from about 8 tokens/s to about 10-12 tokens/s. The useful operational shape is:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
```

The fork will continue to track upstream llama.cpp changes. Keep these private switches small, opt-in, and easy to rebase.

Telemetry showed that the MacBook RPC shard is on the token-generation critical path. `GRAPH_RECOMPUTE` is already fire-and-forget from the client, while the following `get_tensor tensor=norm` waits about 52-53 ms for only 20480 bytes. That wait matches the MacBook Metal recompute time of about 50 ms. This means the bottleneck is dependency and server-completion time, not network bandwidth or client-side graph command blocking.

Do not implement `GGML_RPC_ASYNC_WORKER=1` for this topology. It would move the wait to a future that the CUDA0 output split immediately needs. Future work should focus on placement, context or KV memory pressure, model choice, or topology changes.

## Task 1: RPC Cache Scope

Status: implemented in the current working tree.

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `ggml/include/ggml-rpc.h`
- `notes/rpc/04-data-movement-and-kv-cache.md`
- `notes/rpc/05-performance-bottlenecks.md`
- `notes/rpc/06-tuning-experiments.md`

Behavior:

- Add `GGML_RPC_CACHE_SCOPE=all|weights|none`.
- Preserve current behavior with the default `all`.
- Use `weights` to hash and cache only weight buffers.
- Use `none` to disable client-side RPC cache use.
- Add `cache_write` intent to `RPC_CMD_SET_TENSOR`.

Verification:

- User ran a CMake build and reported it completed with warnings.
- Run the same model and prompt with `all`, `weights`, and `none`.
- Compare prompt-processing speed and RPC cache write log lines.

## Task 2: Opt-In RPC And Scheduler Telemetry

Status: implemented in the current working tree.

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `ggml/src/ggml-backend.cpp`
- `notes/rpc/06-tuning-experiments.md`
- `notes/rpc/08-launch-command-analysis.md`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`

Behavior:

- Add `GGML_RPC_TELEMETRY=1`.
- In the scheduler split loop, log split index, backend name, input count, copied bytes, copy time, compute time, and node count.
- In the RPC graph client, log full graph send vs graph recompute, graph payload bytes, node count, serialize time, and RPC command time.
- In the RPC server graph handlers, log backend compute time for full graph and recompute commands.
- In the RPC tensor cache path, log large-tensor cache events: skipped by scope, hash hit, hash miss, and write intent.
- In `get_tensor`, log blocking wait time with endpoint, device, tensor name, and byte count.
- Add aggregate `rpc_telemetry: summary ...` totals at normal process exit for scheduler splits, RPC graph client commands, RPC server compute, and cache events.

Verification:

- User runs a CMake build for `ggml-rpc-server`.
- Run a short prompt with `GGML_RPC_TELEMETRY=1`.
- Confirm logs show scheduler splits, RPC graph mode, and cache events.
- Stop the process normally and confirm summary lines print at exit.

## Task 3: Telemetry Experiment Notes

Status: implemented in the current working tree.

Files:

- `notes/rpc/06-tuning-experiments.md`
- `notes/rpc/08-launch-command-analysis.md`

Behavior:

- Add an experiment template for telemetry runs.
- Capture which fields to compare for PP and TG.
- Keep the known `--device CUDA0,CUDA1,RPC0 --tensor-split 7,3,6` launch shape as the first test.

Verification:

- Notes mention `GGML_RPC_TELEMETRY=1`.
- Notes include the cache-scope matrix.

## Task 4: RPC-Aware Placement Scoring

Status: implemented in the current working tree.

Files:

- `src/llama-model.cpp`
- `common/fit.cpp`
- `notes/rpc/06-tuning-experiments.md`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`

Behavior:

- Add `LLAMA_RPC_SPLIT_SCALE`.
- Preserve default placement when the knob is unset.
- Scale RPC reported free memory only for automatic placement.
- Leave explicit `--tensor-split` unchanged.

Verification:

- Compare automatic placement with and without `LLAMA_RPC_SPLIT_SCALE`.
- Confirm the model still fits.
- Compare PP and TG against the manual `--tensor-split` baseline.

## Task 5: Output Placement Controls

Status: implemented in the current working tree as a minimal private-fork experiment.

Files:

- `src/llama-model.cpp`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`

Behavior:

- Add `LLAMA_OUTPUT_LAYER_DEVICE`.
- Preserve default output placement when the env var is unset or empty.
- Match the env var against selected offload device names from `--device`.
- Override only the synthetic output layer placement, which covers final norm plus the output LM head.
- Do not move the last transformer block, repeating layer split, KV placement, tensor split math, `n_gpu_layers`, or CLI arguments.
- Fail fast when the env var names an unknown selected offload device.

Launch example:

```text
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0 --rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 7,3,6
```

Verification:

- Compare output backend placement in logs with and without `LLAMA_OUTPUT_LAYER_DEVICE=CUDA0`.
- Confirm repeating layer assignment remains unchanged for `--tensor-split 7,3,6`.
- Compare TG speed and boundary-copy bytes.
- Confirm an unknown env value exits with a message listing the selected offload devices.

## Task 6: RPC Pipeline Diagnostic

Status: implemented and evaluated as a diagnostic.

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`
- `notes/rpc/06-tuning-experiments.md`

Behavior:

- Add `GGML_RPC_PIPELINE=1`.
- Default off: RPC keeps reporting no async and no events.
- When enabled, RPC reports async and events so llama.cpp can enable scheduler pipeline parallelism.
- Implement only client-side no-op RPC events because RPC commands are still blocking.
- Do not change RPC protocol, server behavior, transport, worker threads, or async tensor copy yet.
- Treat this as a diagnostic for scheduler pipeline behavior and memory pressure, not as real async RPC.

Verification:

- User runs the relevant build targets.
- Run the same prompt with and without `GGML_RPC_PIPELINE=1`.
- Confirm startup says `pipeline parallelism enabled` when the switch is on.
- Compare PP/TG, `rpc_telemetry: summary sched ...`, and whether graph reservation falls back after extra compute-buffer pressure.
- The diagnostic did not materially improve TG and increased memory pressure. Keep it off for normal use.

## Task 7: `get_tensor` Wait Telemetry

Status: implemented in the current working tree.

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`
- `notes/rpc/11-async-rpc-research-report.md`

Behavior:

- Under `GGML_RPC_TELEMETRY=1`, log `get_tensor` wait time as `wait_ms`.
- Include RPC endpoint, device id, tensor name, and byte count.
- Leave `graph_compute` telemetry unchanged because recompute and full graph paths already log `cmd_ms`.
- Do not change RPC protocol, scheduler behavior, cache policy, events, or async behavior.

Verification:

- User built and ran the telemetry branch.
- Client logs showed `graph client ... mode=recompute ... cmd_ms=0.001-0.002`.
- Client logs showed `get_tensor client endpoint=192.168.0.118:50052 device=0 tensor=norm bytes=20480 wait_ms=52-53`.
- Server logs showed `graph server mode=recompute device=0 backend=MTL0 nodes=1369 compute_ms=about 50`.
- This confirms the token-generation wait lands at the dependent RPC0 to CUDA0 output boundary.

## Stop Point

Keep the current private-fork changes and continue to take upstream llama.cpp changes. The next useful work is operational tuning, not more RPC execution machinery:

- Try to reduce RPC0's critical-path layer share if memory allows.
- Explore KV/cache memory reduction if it lets more work stay on CUDA.
- Treat `--split-mode row` or tensor-parallel modes as a controlled falsification test only; over 1 GbE they are likely to add more synchronization.
- Do not pursue client-side async worker work unless a new placement or scheduler structure creates independent CUDA work that can overlap with RPC0.
