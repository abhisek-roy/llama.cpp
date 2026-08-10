# RPC Performance System Design

Traceability source commit: `59abc3967`

This page captures the system design before more RPC performance changes. The goal is to make the moving parts visible, then break the work into small private-fork changes.

## Design Goals

- Keep the current RPC model: the main llama.cpp process owns inference, the RPC server exposes remote ggml devices.
- Make performance visible before changing placement or execution behavior.
- Keep private-fork changes small enough to understand and test.
- Prefer controls that preserve current behavior by default.
- Optimize for the observed setup first: Linux host with `CUDA0`, `CUDA1`, and MacBook `RPC0` over wired 1 GbE.

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
    F -->|RPC| H["RPC graph command"]
    H --> I["server reconstructs graph fragment"]
    I --> J["remote backend compute"]
    J --> K["RPC command returns"]
    G --> L["next split"]
    K --> L
```

The important behavior is that split boundaries can move activations between devices. With RPC, a boundary can become a network transfer. The current RPC backend does not expose async events, so the scheduler cannot fully overlap remote and local pipeline stages.

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

## Telemetry Design

The next design step should add timing data before changing placement logic. The telemetry should answer what each split costs and where time moves when flags change.

```mermaid
flowchart TD
    A["llama_context::process_ubatch"] --> B["graph build or graph reuse"]
    B --> C["ggml_backend_sched_compute_splits"]
    C --> D["copy split inputs"]
    C --> E["compute split"]
    E --> F{"backend is RPC?"}
    F -->|no| G["local compute time"]
    F -->|yes| H["ggml_backend_rpc_graph_compute"]
    H --> I["send graph or recompute command"]
    I --> J["server backend compute"]
    J --> K["receive completion"]
```

Useful fields:

- phase: prompt processing or token generation
- ubatch size and graph reuse state
- split index and backend name
- copied bytes into each split
- copy time
- backend compute time
- RPC command time
- RPC graph mode: full graph send or graph recompute
- RPC cache event: disabled, skipped by scope, hash miss, hash hit, cache write

The first telemetry version should be opt-in and log-only. It should not change scheduling behavior.

`GGML_RPC_TELEMETRY=1` also keeps aggregate counters and prints `rpc_telemetry: summary ...` lines when the process exits normally. These summaries group scheduler split time by backend, RPC graph client time by backend and graph mode, RPC server compute time by backend and graph mode, and cache traffic by side, event, and usage. They are process totals, not per request totals. If the process is killed before normal shutdown, summarize the raw telemetry log with the bash commands in [Tuning experiments](06-tuning-experiments.md).

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

## Future Async And Round-Trip Work

```mermaid
flowchart LR
    subgraph Current["Current RPC execution"]
        A1["send command"] --> A2["wait"]
        A2 --> A3["server compute"]
        A3 --> A4["return"]
        A4 --> A5["next split"]
    end

    subgraph Future["Future async direction"]
        B1["submit remote work"] --> B2["local work overlaps"]
        B1 --> B3["remote work runs"]
        B2 --> B4["event sync when needed"]
        B3 --> B4
    end
```

Async RPC copy, compute, and events are likely the real architectural win for token generation. They are also the largest change because they affect backend capabilities, scheduler assumptions, protocol behavior, and error handling. They should come after telemetry and smaller placement work.

## Roadmap

```mermaid
flowchart TD
    A["Capture system design in notes"] --> B["Run cache-scope experiments"]
    B --> C["Add RPC and scheduler telemetry"]
    C --> D["Use telemetry to tune tensor split manually"]
    D --> E["Add RPC-aware placement scoring"]
    E --> F["Evaluate output placement controls"]
    C --> G["Decide whether async RPC is worth the cost"]
    G --> H["Prototype async copy, compute, and events"]
    H --> I["Reduce per-token RPC round trips"]
```

Best first implementation changes:

1. Add RPC and scheduler timing telemetry.
2. Finish and test RPC cache policy.
3. Add RPC-aware placement scoring after telemetry.

Bigger changes:

4. Implement async RPC copy, compute, and events.
5. Reduce per-token RPC round trips.
6. Add explicit output and final-layer placement controls.

## Immediate Experiment Plan

Use the current useful launch shape:

```text
--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 7,3,6
```

Compare the same prompt and model with the MacBook server using `--cache`:

```text
GGML_RPC_CACHE_SCOPE=all
GGML_RPC_CACHE_SCOPE=weights
GGML_RPC_CACHE_SCOPE=none
```

Expected interpretation:

- `all`: preserves old cache behavior and can still write transient large tensors.
- `weights`: should keep warm model-load cache benefits and reduce prompt-time cache writes.
- `none`: should remove client-side cache probes and writes, but warm model load can become slower.

Record prompt processing speed, token generation speed, RPC cache log lines, and whether this is the first or second run after the server cache is warm.

After that, test automatic placement with `LLAMA_RPC_SPLIT_SCALE`:

```text
LLAMA_RPC_SPLIT_SCALE=0.75
LLAMA_RPC_SPLIT_SCALE=0.50
LLAMA_RPC_SPLIT_SCALE=0.25
```

Run without explicit `--tensor-split` for this experiment. Compare the chosen layer placement against the manual `--tensor-split 7,3,6` baseline.
