# RPC Performance System Design

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

This page captures the current RPC performance design after merging upstream's queued dispatcher and real event support. It separates current architecture from measurements made with the older direct-call implementation.

## Design Goals

- Keep the current RPC model: the main llama.cpp process owns inference, the RPC server exposes remote ggml devices.
- Make performance visible before changing placement or execution behavior.
- Keep private-fork changes small enough to understand and test.
- Prefer controls that preserve current behavior by default.
- Optimize for the observed setup first: Linux host with `CUDA0`, `CUDA1`, and MacBook `RPC0` over wired 1 GbE.

## Current Private Fork Outcome

Before the upstream merge, this investigation reached a useful private-fork stop point. The setup was worth continuing to use, especially for long-context runs where the MacBook RPC device helps the model fit. Prompt processing was good enough for the target workflow, and token generation improved from about 8 tokens/s to about 10-12 tokens/s with the tested placement and cache changes. These results now need a post-merge comparison.

Current post-merge validation candidate:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
```

Useful environment:

```text
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
```

The private switches remain operational controls and diagnostics while this fork continues to take upstream llama.cpp changes. Keep them small and easy to rebase. Do not treat them as an upstream-ready feature set without a separate design pass.

The private `GGML_RPC_PIPELINE` diagnostic no longer exists. Upstream now provides real asynchronous graph and tensor submission plus queue-backed events. The fork retains cache scope, telemetry, RPC split scaling, and output placement.

The main performance conclusion is that `RPC0` is useful for capacity but slow as a required token-generation stage. Telemetry showed the MacBook Metal recompute takes about 50 ms, and the Linux client then waits about 52-53 ms in `get_tensor` for a tiny `norm` tensor of 20480 bytes. That wait is dependency and server-completion time, not network bandwidth.

## Current Runtime Shape

```mermaid
flowchart LR
    subgraph MainHost["Linux main host"]
        App["llama-server"]
        Loader["model loader"]
        Sched["ggml scheduler"]
        CUDA0["CUDA0: RTX 4070 Ti SUPER"]
        CUDA1["CUDA1: RTX 2060 SUPER"]
        RPCClient["RPC backend client"]
    end

    subgraph RemoteHost["MacBook RPC host"]
        RPCServer["ggml-rpc-server"]
        Metal["MTL0: Metal backend"]
        Cache["optional tensor cache"]
    end

    App --> Loader
    App --> Sched
    Loader --> CUDA0
    Loader --> CUDA1
    Loader --> RPCClient
    Sched --> CUDA0
    Sched --> CUDA1
    Sched --> RPCClient
    RPCClient <-->|RPC protocol over LAN| RPCServer
    RPCServer --> Metal
    RPCServer --> Cache
```

RPC is not a remote llama server. The MacBook does not own tokenization, sampling, context state, or the whole model graph. It owns remote buffers and runs graph fragments assigned to its exposed backend.

## Performance Phases

```mermaid
flowchart TD
    A["Start or request"] --> B{"Phase"}

    B --> C["Model load"]
    C --> C1["allocate buffers"]
    C1 --> C2["set weight tensors"]
    C2 --> C3["optional RPC cache hash or write"]

    B --> D["Prompt processing"]
    D --> D1["large ubatches"]
    D1 --> D2["scheduler split execution"]
    D2 --> D3["activation copies across boundaries"]
    D3 --> D4["RPC graph compute for remote splits"]

    B --> E["Token generation"]
    E --> E1["small ubatches, often one new token"]
    E1 --> E2["same split and copy structure"]
    E2 --> E3["latency and synchronization dominate"]
```

Prompt processing can hide some RPC cost because it has more work per ubatch. Token generation repeats the same distributed execution path with much less work per step, so fixed RPC costs hurt more.

## RPC Cache Policy

`ggml-rpc-server --cache` is server capability. It means the server has a cache directory and can read or write cached tensor files.

`GGML_RPC_CACHE_SCOPE` is client policy. It decides which client tensor uploads should use the cache protocol.

```mermaid
sequenceDiagram
    participant Client as RPC client
    participant Server as RPC server
    participant Cache as Server cache
    participant Device as Remote backend

    Client->>Client: check size and GGML_RPC_CACHE_SCOPE
    alt cache is allowed for this buffer
        Client->>Server: RPC_CMD_SET_TENSOR_HASH
        Server->>Cache: lookup hash
        alt cache hit
            Cache-->>Server: bytes
            Server->>Device: ggml_backend_tensor_set
            Server-->>Client: hit
        else cache miss
            Server-->>Client: miss
            Client->>Server: RPC_CMD_SET_TENSOR with cache_write=1 and bytes
            Server->>Cache: write bytes if --cache is enabled
            Server->>Device: ggml_backend_tensor_set
        end
    else cache is not allowed for this buffer
        Client->>Server: RPC_CMD_SET_TENSOR with cache_write=0 and bytes
        Server->>Device: ggml_backend_tensor_set
    end
```

Use `GGML_RPC_CACHE_SCOPE=weights` first with `ggml-rpc-server --cache`. This keeps warm model-weight loads but stops transient inference tensors from doing cache hash checks and cache file writes. Use `none` as a diagnostic or when the server does not use `--cache`.

## Scheduler Split Execution

```mermaid
flowchart TD
    A["ggml graph"] --> B["assign each node to a backend"]
    B --> C["split graph at backend boundaries"]
    C --> D["create copy tensors for split inputs"]
    D --> E["compute splits in order"]

    E --> F{"split backend"}
    F -->|CUDA| G["local backend compute"]
    F -->|RPC| H["enqueue RPC graph command"]
    H --> I["dispatcher sends in queue order"]
    I --> J["server reconstructs graph fragment"]
    J --> K["remote backend compute"]
    G --> L["next split"]
    K --> M["dependent read or event completes"]
    M --> L
```

The important behavior is that split boundaries can move activations between devices. With RPC, a boundary can become a network transfer. The dispatcher and events allow overlap when independent work exists, but a dependent next split still waits for the remote result.

## Why Token Generation Is Harder

```mermaid
flowchart LR
    subgraph PP["Prompt processing"]
        PP1["many tokens per ubatch"] --> PP2["more compute per RPC command"]
        PP2 --> PP3["overhead is amortized"]
    end

    subgraph TG["Token generation"]
        TG1["one or few new tokens"] --> TG2["same boundary and command costs"]
        TG2 --> TG3["latency is paid often"]
    end
```

This is why a split can improve long prompt ingestion but still reduce interactive token generation speed. The useful split depends on whether the workload is prompt-heavy, generation-heavy, or mixed.

## Telemetry

Telemetry is implemented and opt-in. It answers what each split costs and where time moves when flags change.

```mermaid
flowchart TD
    A["llama_context::process_ubatch"] --> B["graph build or graph reuse"]
    B --> C["ggml_backend_sched_compute_splits"]
    C --> D["copy split inputs"]
    C --> E["compute split"]
    E --> F{"backend is RPC?"}
    F -->|no| G["local compute time"]
    F -->|yes| H["ggml_backend_rpc_graph_compute"]
    H --> I["enqueue graph or recompute command"]
    I --> J["dispatcher sends command"]
    J --> K["server backend compute"]
    K --> L["later dependency waits if needed"]
```

Useful fields:

- phase: prompt processing or token generation
- ubatch size and graph reuse state
- split index and backend name
- copied bytes into each split
- copy time
- backend compute time
- RPC graph serialization and enqueue time
- RPC graph mode: full graph send or graph recompute
- RPC cache event: disabled, skipped by scope, hash miss, hash hit, cache write

`GGML_RPC_TELEMETRY=1` is log-only. It also keeps aggregate counters and prints `rpc_telemetry: summary ...` lines when the process exits normally. Graph-client summaries now report `enqueue_ms`, because asynchronous submission does not include transfer or remote compute time. Server `compute_ms` and a later dependent `get_tensor wait_ms` remain the authoritative indicators of the critical path. These summaries are process totals, not per-request totals. If the process is killed before normal shutdown, summarize the raw telemetry log with the bash commands in [Tuning experiments](06-tuning-experiments.md).

## Placement Scoring Design

Current automatic split behavior is mostly memory based. For RPC, memory is not enough because a remote device has extra latency and network bandwidth cost.

```mermaid
flowchart TD
    A["candidate device"] --> B["free memory"]
    A --> C["estimated compute speed"]
    A --> D["network latency and bandwidth"]
    A --> E["backend class: local or RPC"]
    A --> F["output layer preference"]

    B --> G["placement score"]
    C --> G
    D --> G
    E --> G
    F --> G

    G --> H["layer ranges and tensor split"]
```

A first private-fork version should not try to solve all placement. It can start with an RPC penalty or speed weight that biases fewer layers to `RPC0` while still fitting memory. Telemetry should come first so the penalty can be based on observed cost instead of guesswork.

The initial private-fork switch is `LLAMA_RPC_SPLIT_SCALE`. It is disabled by default. When unset, automatic split behavior uses reported free memory exactly as before. When set to a positive value, RPC devices count as `reported_free_memory * LLAMA_RPC_SPLIT_SCALE` for automatic placement. Values below `1.0` penalize RPC devices; for example `0.5` treats the MacBook RPC device as if it had half of its reported free memory. Explicit `--tensor-split` values are not changed by this switch.

## Output Layer Device Override

The private-fork output placement switch is `LLAMA_OUTPUT_LAYER_DEVICE`. It is disabled when unset or empty. When set, it must match one selected offload device name from `--device`, for example `CUDA0`. Unknown names fail fast and list the selected offload devices.

This is output-only placement: final norm plus output LM head. It does not move the last transformer block, the repeating layer split, KV placement, tensor split math, `n_gpu_layers`, or CLI arguments. The default output assignment still uses the synthetic layer after the repeating layers; the override only replaces the output buffer-type list with the matched device's existing list.

Use it with the existing manual split experiment:

```text
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0 --rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 7,3,6
```

Expected verification:

- Logs show the default output device and then `LLAMA_OUTPUT_LAYER_DEVICE=CUDA0 sets output layer device = CUDA0`.
- Repeating layer assignment remains unchanged for the same `--tensor-split 7,3,6`.
- TG speed and telemetry boundary-copy bytes can be compared against the same launch without the env var.

Risk: moving output weights to `CUDA0` increases local GPU memory use. It can also add a boundary copy from the device that owns the last repeating layer to `CUDA0`; the experiment is useful only if avoiding RPC output placement is worth that cost.

## Current Async And Round-Trip Model

```mermaid
flowchart LR
    subgraph Current["Queued RPC execution"]
        A1["submit remote work"] --> A2["dispatcher queue"]
        A2 --> A3["server compute"]
        A1 --> A4["independent scheduler work"]
        A3 --> A5["event or dependent read"]
        A4 --> A5
    end
```

The upstream dispatcher now queues graph compute, graph recompute, and async tensor set/get operations. Synchronize and event operations are real queue fences. This is the client-side worker architecture that the earlier design treated as future work.

The measured topology still has a hard dependency: the CUDA0 output split needs RPC0's result. Async submission can move the wait to a later read or event, but cannot remove remote compute time. The pre-merge fake-event `GGML_RPC_PIPELINE=1` result remains historical evidence only; that switch and its no-op events were removed.

Async tensor copy and graph plans are still absent. Consider them only after post-merge telemetry shows an avoidable serialization point with useful independent work available.

## Protocol And Transport

This fork uses RPC protocol `7.0.0` because `cache_write` changes the `RPC_CMD_SET_TENSOR` payload. Use matching client and server binaries.

TCP remains the fallback transport. Compatible builds auto-negotiate Linux RoCE/InfiniBand or Apple silicon RDMA over Thunderbolt. Set `GGML_RPC_NO_RDMA=1` to force TCP for a controlled comparison.

## Consolidated Roadmap State

```mermaid
flowchart TD
    A["Merge upstream dispatcher and events"] --> B["Build RPC targets"]
    B --> C["Update architecture notes"]
    C --> D["Run post-merge telemetry"]
    D --> E["Compare placement and transport"]
    E --> F["Continue operational tuning"]
```

Completed private-fork changes:

1. RPC cache scope.
2. RPC and scheduler timing telemetry.
3. RPC-aware placement scale for automatic placement.
4. Output placement override.
5. `get_tensor` wait telemetry.
6. Upstream queued dispatcher, async tensor operations, and real events.
7. Linux and Apple RDMA transport support from upstream.

Historical and removed:

1. `GGML_RPC_PIPELINE` fake-event diagnostic. Results remain in the research notes but the code was removed.

Deferred work:

1. Async tensor copy and graph plans, pending post-merge evidence.
2. Further placement and memory work. This remains the likely useful direction if more TG speed is needed.
3. Transport benchmarking on hardware that can use RDMA.

## Operational Follow-Up Checks

Use the current useful launch shape:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5
```

The normal-use setting is:

```text
GGML_RPC_CACHE_SCOPE=weights
LLAMA_OUTPUT_LAYER_DEVICE=CUDA0
```

Use `GGML_RPC_TELEMETRY=1` only for measurement runs. Normal runs can leave telemetry off.

Optional diagnostics:

- `GGML_RPC_CACHE_SCOPE=none`: use only if cache telemetry suggests unexpected hash or write behavior.
- `LLAMA_RPC_SPLIT_SCALE`: use only when testing automatic placement without explicit `--tensor-split`.
- `GGML_RPC_NO_RDMA=1`: force TCP when comparing it with an automatically negotiated RDMA link.

For placement experiments with `LLAMA_RPC_SPLIT_SCALE`, run without explicit `--tensor-split` and compare the chosen layer placement against the manual `--tensor-split 6.5,3,6.5` baseline:

```text
LLAMA_RPC_SPLIT_SCALE=0.75
LLAMA_RPC_SPLIT_SCALE=0.50
LLAMA_RPC_SPLIT_SCALE=0.25
```
