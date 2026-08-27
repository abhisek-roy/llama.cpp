# RPC Performance Implementation Plan

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

This document records the private-fork implementation plan and its current status after the 2026-08-27 upstream merge. Completed items are retained for traceability. Superseded items are labeled rather than rewritten as if they were current work.

## Ground Rules

- Keep each change opt-in unless it is preserving existing behavior.
- Do not change placement behavior before telemetry exists.
- Do not commit or push from an agent session.
- The user runs CMake build commands.
- Prefer notes updates with each code change so the system model stays current.

## Final Investigation Status

The pre-merge private-fork investigation was complete enough for continued use. That setup gave good prompt processing for the target workflow and improved token generation from about 8 tokens/s to about 10-12 tokens/s. Use the following shape as the first post-merge validation candidate:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
```

The fork will continue to track upstream llama.cpp changes. Keep the remaining private switches small, opt-in, and easy to rebase.

Telemetry showed that the MacBook RPC shard is on the token-generation critical path. `GRAPH_RECOMPUTE` is already fire-and-forget from the client, while the following `get_tensor tensor=norm` waits about 52-53 ms for only 20480 bytes. That wait matches the MacBook Metal recompute time of about 50 ms. This means the bottleneck is dependency and server-completion time, not network bandwidth or client-side graph command blocking.

Upstream now supplies the equivalent worker architecture through `rpc_dispatcher`, async tensor and graph submission, synchronization, and real events. This should be measured before considering more async machinery. The dependent CUDA0 output split can still move the wait to a later fence without reducing the MacBook's compute time.

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
- In the RPC graph client, log full graph send vs graph recompute, graph payload bytes, node count, serialize time, and queue submission time.
- In the RPC server graph handlers, log backend compute time for full graph and recompute commands.
- In the RPC tensor cache path, log large-tensor cache events: skipped by scope, hash hit, hash miss, and write intent.
- In `get_tensor`, log blocking wait time with endpoint, device, tensor name, and byte count.
- Add aggregate `rpc_telemetry: summary ...` totals at normal process exit for scheduler splits, RPC graph client enqueue time, RPC server compute, and cache events.

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

Status: historical experiment, removed during the upstream merge.

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `notes/rpc/09-performance-system-design.md`
- `notes/rpc/10-implementation-plan.md`
- `notes/rpc/06-tuning-experiments.md`

Historical behavior:

- Add `GGML_RPC_PIPELINE=1`.
- Default off: RPC keeps reporting no async and no events.
- When enabled, RPC reports async and events so llama.cpp can enable scheduler pipeline parallelism.
- Implement only client-side no-op RPC events because RPC commands are still blocking.
- Do not change RPC protocol, server behavior, transport, worker threads, or async tensor copy yet.
- Treat this as a diagnostic for scheduler pipeline behavior and memory pressure, not as real async RPC.

Historical result:

- User runs the relevant build targets.
- Run the same prompt with and without `GGML_RPC_PIPELINE=1`.
- Confirm startup says `pipeline parallelism enabled` when the switch is on.
- Compare PP/TG, `rpc_telemetry: summary sched ...`, and whether graph reservation falls back after extra compute-buffer pressure.
- The diagnostic did not materially improve TG and increased memory pressure. Keep it off for normal use.

Current replacement:

- Upstream's `rpc_dispatcher` queues work on a worker thread.
- Graph compute, graph recompute, and backend async tensor operations use `send_async()`.
- Synchronization and events use real queue completion fences.
- RPC reports async and event support without an environment switch.
- `GGML_RPC_PIPELINE` is no longer recognized.

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
- Report graph submission as `enqueue_ms`; it does not include transfer or remote compute.
- Keep `get_tensor wait_ms` as the measurement for a blocking dependent read.

Verification:

- User built and ran the telemetry branch.
- Pre-merge client logs showed `graph client ... mode=recompute ... cmd_ms=0.001-0.002`.
- Client logs showed `get_tensor client endpoint=192.168.0.118:50052 device=0 tensor=norm bytes=20480 wait_ms=52-53`.
- Server logs showed `graph server mode=recompute device=0 backend=MTL0 nodes=1369 compute_ms=about 50`.
- This confirms the token-generation wait lands at the dependent RPC0 to CUDA0 output boundary.

## Task 8: Upstream RPC Integration

Status: code merged and RPC targets built successfully; runtime validation remains.

Behavior:

- Adopt upstream's per-endpoint queued dispatcher.
- Adopt asynchronous graph compute, graph recompute, and tensor set/get.
- Adopt real queue-backed events and unconditional async/event capabilities.
- Remove the private fake-event `GGML_RPC_PIPELINE` implementation.
- Apply `GGML_RPC_CACHE_SCOPE` to both synchronous and asynchronous tensor upload paths.
- Retain `GGML_RPC_TELEMETRY`, `LLAMA_RPC_SPLIT_SCALE`, and `LLAMA_OUTPUT_LAYER_DEVICE`.
- Use protocol `7.0.0` because the private `cache_write` byte changes the tensor payload.
- Document Linux RDMA and Apple silicon RDMA over Thunderbolt.

Verification:

- The pre-merge RPC-enabled build succeeded.
- After conflict resolution, the user successfully built `ggml-rpc-server` and `llama-server`.
- Run a matched client/server smoke test before finalizing the merge.
- Run telemetry once to verify `enqueue_ms`, server `compute_ms`, dependent `wait_ms`, cache scope, and pipeline behavior.

## Stop Point

Keep the current private-fork changes and continue to take upstream llama.cpp changes. Validate the merged async implementation first, then return to operational tuning:

- Try to reduce RPC0's critical-path layer share if memory allows.
- Explore KV/cache memory reduction if it lets more work stay on CUDA.
- Treat `--split-mode row` or tensor-parallel modes as a controlled falsification test only; over 1 GbE they are likely to add more synchronization.
- Do not add another client-side async layer unless post-merge telemetry identifies independent work that the dispatcher cannot overlap.
