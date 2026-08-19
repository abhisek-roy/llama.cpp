# Inference Execution

Traceability source commit: `52e5dda62`

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

### Tensor Use Counts In The Serialized Graph

Since upstream `af5172627` (merged at `52e5dda62`), each serialized graph fragment also carries a per-tensor use count: how many ops in the full graph read that tensor.

- `ggml_visit_parents_graph()` fills the cgraph `visited_hash_set` and `use_counts` at graph build time in the main process.
- Scheduler splits are `ggml_graph_view()` of the full graph, so they share those arrays.
- When serializing a split, `add_tensor()` looks each tensor up in the shared hash set and packs the count into `rpc_tensor.use_count`, the field that used to be 4 bytes of padding. Wire size is unchanged.
- `rpc_server::graph_compute()` restores the counts into the deserialized remote graph.

The counts are sent, not recomputed on the server. The server only sees a partial graph, so a locally recomputed count would be wrong: a tensor used once remotely and once locally would count as 1 there, which breaks the used-exactly-once checks that backend fusion logic relies on (`ggml_node_has_n_uses()` in `ggml/src/ggml-impl.h`).

Status at this merge: no in-tree Metal or CPU fusion rule consumes use counts yet. Current consumers are the meta-backend MoE AllReduce delay logic and OpenVINO. This is infrastructure, so expect no perf change on the dense model.

## Pipeline Parallelism And RPC

llama.cpp has a pipeline-parallel path for layer split mode, but it requires async compute and events on all non-CPU devices. The RPC backend currently reports no async/event support. That means pipeline parallelism is effectively disabled when RPC devices participate.

This is a key reason token generation can be slow over RPC. The system can place layers remotely, but it cannot overlap remote and local stages the way a lower-latency multi-GPU setup might.
