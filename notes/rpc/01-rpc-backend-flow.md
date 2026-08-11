# RPC Backend Flow

Traceability source commit: `b89f44654161`

## What RPC Is

The RPC backend lets one llama.cpp process use devices that live in another process, possibly on another host. The remote process is `ggml-rpc-server`. It exposes local ggml backend devices such as CUDA, Metal, or CPU through a binary RPC protocol.

The main process still controls:

- command-line parsing
- model loading
- tensor placement decisions
- graph construction
- backend scheduling
- sampling and output handling

The remote server controls:

- remote backend initialization
- remote buffer allocation
- remote tensor writes and reads
- remote graph execution for graph fragments assigned to that remote backend

## Server Startup

`tools/rpc/rpc-server.cpp` starts the standalone server. It loads all compiled backends, selects devices, resolves `ggml_backend_rpc_start_server()`, and starts listening.

```mermaid
flowchart TD
    A["ggml-rpc-server starts"] --> B["ggml_backend_load_all"]
    B --> C["Parse --host, --port, --device, --cache"]
    C --> D["Select exposed devices"]
    D --> E["Find RPC backend registry"]
    E --> F["Resolve ggml_backend_rpc_start_server"]
    F --> G["Initialize one backend per exposed device"]
    G --> H["Listen for client connection"]
    H --> I["HELLO protocol handshake"]
    I --> J["Dispatch RPC_CMD_* requests"]
```

Important server controls:

- `--host`: bind address. Keep this on a trusted network only.
- `--port`: listen port, default `50052`.
- `--device`: expose only selected devices.
- `--cache`: enable local tensor cache for large model tensors.
- `--threads`: CPU backend thread count if CPU is exposed.

## Client Registration

The main process adds RPC devices when `--rpc` is parsed.

```mermaid
sequenceDiagram
    participant CLI as llama.cpp main process
    participant Args as common/arg.cpp
    participant RPC as RPC backend registry
    participant Server as ggml-rpc-server

    CLI->>Args: parse --rpc host:port
    Args->>RPC: find backend registry named RPC
    Args->>RPC: call ggml_backend_rpc_add_server(endpoint)
    RPC->>Server: RPC_CMD_DEVICE_COUNT
    Server-->>RPC: remote device count
    RPC-->>Args: registry RPC[endpoint]
    Args->>CLI: register remote devices
```

After registration, RPC devices look like normal ggml GPU devices to the model loader and scheduler. This is why the same multi-GPU flags also affect RPC devices.

## Protocol Shape

The RPC protocol is binary. A request sends:

```text
cmd byte | request size | request payload
```

Some commands receive:

```text
response size | response payload
```

The important command groups are:

- device queries: device count, memory, alignment, max buffer size
- buffer lifecycle: allocate, free, clear
- tensor transfer: set tensor, get tensor, copy tensor
- graph execution: graph compute, graph recompute
- cache support: set tensor hash for large tensors

## What Stays Remote

Remote backend buffers stay in the server process. The client holds remote pointer handles and uses them only as identifiers in later RPC messages.

```mermaid
flowchart LR
    ClientTensor["Client ggml_tensor metadata"] --> Handle["remote_ptr handle"]
    Handle --> RPCMsg["RPC tensor record"]
    RPCMsg --> ServerBuffer["Server backend buffer"]
    ServerBuffer --> DeviceMemory["Remote GPU or unified memory"]
```

The client serializes tensor metadata. The server reconstructs temporary ggml tensors around remote buffers and executes backend operations locally.

## Graph Reuse

For graph execution, the first run sends the serialized graph. If the graph UID matches the previous graph for that remote device, later runs can send `RPC_CMD_GRAPH_RECOMPUTE`.

```mermaid
stateDiagram-v2
    [*] --> NoGraph
    NoGraph --> GraphSent: RPC_CMD_GRAPH_COMPUTE
    GraphSent --> Recompute: same graph uid
    Recompute --> Recompute: RPC_CMD_GRAPH_RECOMPUTE
    GraphSent --> GraphSent: new graph uid
```

This matters for token generation. Reusing the graph avoids resending all graph metadata, but it does not remove network round trips or cross-device tensor copies.
