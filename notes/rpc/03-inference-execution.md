# Inference Execution

Traceability source commit: `b89f44654161`

## Main Flow

The main inference path goes through `llama_decode()`, batch splitting, graph construction or reuse, scheduler allocation, and backend graph execution.

```mermaid
flowchart TD
    A["User batch"] --> B["llama_context::decode"]
    B --> C["memory->init_batch creates ubatch"]
    C --> D["process_ubatch"]
    D --> E{"Can reuse previous graph?"}
    E -->|yes| F["Reuse graph topology"]
    E -->|no| G["model.build_graph"]
    G --> H["ggml_backend_sched_alloc_graph"]
    F --> I["Set input tensors"]
    H --> I
    I --> J["ggml_backend_sched_graph_compute_async"]
    J --> K["Scheduler computes backend splits"]
    K --> L["Collect logits or sampling result"]
```

Prompt processing and token generation use the same machinery, but they have different shapes:

- Prompt processing uses larger ubatches. There is more compute per scheduling and RPC round trip.
- Token generation often processes one new token per sequence. There is less compute per round trip, so latency dominates more easily.

## Scheduler Splits

The ggml backend scheduler assigns each graph node to a backend. Weight buffers strongly influence placement: an operation with a weight tends to run on the backend that owns that weight.

```mermaid
flowchart TD
    A["Graph nodes"] --> B["Assign preallocated tensors"]
    B --> C["Prefer backend that owns weight buffer"]
    C --> D["Expand non-CPU assignments through adjacent ops"]
    D --> E["Assign remaining ops to compatible backend"]
    E --> F["Split graph at backend boundaries"]
    F --> G["Create tensor copies for cross-backend inputs"]
    G --> H["Compute each split"]
```

When a split needs data from another backend, the scheduler inserts a copy into the destination backend. With RPC, these copies are especially important because they can become network transfers.

## RPC Graph Compute

When a scheduler split is assigned to an RPC backend, the RPC backend serializes the graph fragment and sends it to the remote server.

```mermaid
sequenceDiagram
    participant Main as Main process
    participant Sched as ggml scheduler
    participant RPC as RPC backend
    participant Server as ggml-rpc-server
    participant Device as Remote backend

    Main->>Sched: compute graph
    Sched->>Sched: split graph by backend
    Sched->>RPC: compute RPC split
    RPC->>Server: RPC_CMD_GRAPH_COMPUTE or RPC_CMD_GRAPH_RECOMPUTE
    Server->>Server: reconstruct graph from rpc_tensor metadata
    Server->>Device: ggml_backend_graph_compute
    Device-->>Server: completion
    Server-->>RPC: command complete
    RPC-->>Sched: split complete
    Sched-->>Main: graph complete
```

The remote server does not sample tokens and does not own the full llama context. It only executes the ggml graph fragment it receives.

## Pipeline Parallelism And RPC

llama.cpp has a pipeline-parallel path for layer split mode, but it requires async compute and events on all non-CPU devices. The RPC backend currently reports no async/event support. That means pipeline parallelism is effectively disabled when RPC devices participate.

This is a key reason token generation can be slow over RPC. The system can place layers remotely, but it cannot overlap remote and local stages the way a lower-latency multi-GPU setup might.
