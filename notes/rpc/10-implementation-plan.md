# RPC Performance Implementation Plan

Traceability source commit: `59abc3967`

This plan breaks [RPC performance system design](09-performance-system-design.md) into small private-fork changes. Do not submit these changes upstream without a separate discussion.

## Ground Rules

- Keep each change opt-in unless it is preserving existing behavior.
- Do not change placement behavior before telemetry exists.
- Do not commit or push from an agent session.
- The user runs CMake build commands.
- Prefer notes updates with each code change so the system model stays current.

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

Verification:

- User runs a CMake build for `ggml-rpc-server`.
- Run a short prompt with `GGML_RPC_TELEMETRY=1`.
- Confirm logs show scheduler splits, RPC graph mode, and cache events.

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

Files:

- `src/llama-model.cpp`
- `common/fit.cpp`
- `notes/rpc/09-performance-system-design.md`

Behavior:

- Add a private-fork scoring knob only after telemetry data exists.
- Start with a simple RPC penalty or speed weight.
- Preserve default placement when the knob is unset.

Verification:

- Compare automatic placement with and without the new scoring knob.
- Confirm the model still fits.
- Compare PP and TG against the manual `--tensor-split` baseline.

## Task 5: Output Placement Controls

Files:

- `src/llama-model.cpp`
- `src/llama-context.cpp`
- `notes/rpc/09-performance-system-design.md`

Behavior:

- Explore a local-output preference without reordering all layer ranges.
- Do this only after telemetry shows output placement is a material cost.

Verification:

- Compare output backend placement in logs.
- Compare TG speed and boundary-copy bytes.

## Task 6: Async RPC Prototype

Files:

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
- `ggml/src/ggml-rpc/transport.cpp`
- `ggml/src/ggml-backend.cpp`
- `notes/rpc/09-performance-system-design.md`

Behavior:

- Prototype async copy, compute, and event support for RPC.
- Keep it behind a private fork flag.
- Do not start this until telemetry and placement experiments show that serialized RPC split execution is still the main bottleneck.

Verification:

- Compare split overlap with telemetry.
- Compare TG speed on the same prompt and split.
