# Async RPC Performance Research Report

Private fork of llama.cpp. Scope: improving RPC backend throughput on a Linux
host (CUDA0 + CUDA1) with a MacBook M4 Pro (RPC0, Metal) joined over wired 1 GbE.
No upstream PRs; private fork only.

This report is design-only. No code changes, no commits, no PRs.

---

## 1. Current execution model

### 1.1 Topology and launch

- Host: RTX 4070 Ti SUPER (CUDA0) + RTX 2060 SUPER (CUDA1).
- RPC node: MacBook M4 Pro (Metal) over wired 1 GbE.
- Launch: `--rpc 192.168.0.118:50052 --device CUDA0,CUDA1,RPC0 --tensor-split 6.5,3,6.5`.
- Scheduler has 3 GPU backends plus the CPU backend (required last). Splits are
  formed at every backend boundary in the compute graph.

### 1.2 Scheduler split execution (`ggml_backend_sched_compute_splits`)

File: `ggml/src/ggml-backend.cpp` around line 1700.

For each split `i` in `0..n_splits-1`, **serially**:

1. For each input tensor crossing into the split's backend:
   - If the input is a user `GGML_TENSOR_FLAG_INPUT`, `event_synchronize` or
     `ggml_backend_synchronize` on the split backend, then a blocking
     `ggml_backend_tensor_copy`.
   - Else attempt `cpy_tensor_async` on the destination backend. If the backend
     does not implement it (RPC does not), fall back to
     `ggml_backend_synchronize(input_backend)` + a blocking
     `ggml_backend_tensor_copy` (host bounce via `malloc`/`get`/`set`).
   - After the copy, `event_record` is called on the split backend **only if**
     `sched->events[backend_id][cur_copy] != NULL`.
2. `ggml_backend_graph_compute_async(split_backend, &split->graph)` - which for
   RPC just calls `ggml_backend_rpc_graph_compute` (no async path).
3. Telemetry is logged; loop continues to the next split.

Pipeline parallelism (the `parallel=true` branch of `ggml_backend_sched_new`,
which allocates `GGML_SCHED_MAX_COPIES` event slots and enables overlap) is
**disabled** whenever any non-CPU backend lacks `caps.async` or `caps.events`.
See `src/llama-context.cpp` line ~424-451.

### 1.3 RPC backend capability advertisement

File: `ggml/src/ggml-rpc/ggml-rpc.cpp`, `ggml_backend_rpc_device_get_props`
(line ~2225) and `ggml_backend_rpc_interface` (line ~1071):

```
props->caps.async  = false;
props->caps.events = false;
/* .set_tensor_async       = NULL */
/* .get_tensor_async       = NULL */
/* .cpy_tensor_async       = NULL */
/* .synchronize            = ggml_backend_rpc_synchronize   // no-op */
/* .event_record / event_wait = NULL */
```

Consequences:
- `llama-context.cpp` forces `cparams.pipeline_parallel = false` because RPC
  reports `async=false, events=false`. The scheduler runs with `n_copies=1`
  and **no event slots** (`events[b][c]` are all NULL).
- In `ggml_backend_sched_compute_splits`, every cross-backend copy hits the
  `ggml_backend_synchronize(input_backend)` + blocking copy fallback.
- `ggml_backend_rpc_synchronize` is a **no-op**, so synchronizing the RPC
  backend does nothing - the real synchronization is implicit in the blocking
  `send_rpc_cmd` socket round-trip.

### 1.4 Client connection model

`get_socket(endpoint)` (line ~646) caches **one** `shared_ptr<socket_t>` per
endpoint in a static `unordered_map<string, weak_ptr<socket_t>>`. All RPC
commands for a given endpoint - buffer alloc, set/get tensor, copy, graph
compute - are multiplexed over that single TCP connection.

The mutex inside `get_socket` only guards the map lookup. There is **no mutex
around send/recv** of a command once the socket is obtained. Each `send_rpc_cmd`
is a blocking request/response: send cmd+payload, then (for commands that have
a response) block on `recv_data`. The server loop in `rpc_serve_client` is a
single-threaded `while (true) { recv cmd; switch; }` that processes one command
at a time to completion.

### 1.5 Transport

File: `ggml/src/ggml-rpc/transport.cpp`. `send_data`/`recv_data` are blocking
`send(2)`/`recv(2)` loops on TCP (or blocking RDMA post+poll when
`GGML_RPC_RDMA` is compiled in). No non-blocking I/O, no `poll`/`epoll`/io_uring
on the data path. `MSG_DONTWAIT` is not used.

### 1.6 Graph compute path

Client `ggml_backend_rpc_graph_compute` (line ~1034):
- If `cgraph->uid != 0 && last_graph_uid == uid`, send `RPC_CMD_GRAPH_RECOMPUTE`
  (16-byte request, server replays stored graph).
- Else serialize the graph (device, n_nodes, node ptrs, tensors) and send
  `RPC_CMD_GRAPH_COMPUTE`.

Both block until the server has finished `ggml_backend_graph_compute` on the
remote backend and replied. There is no way to issue a graph and continue
client-side work while it runs.

### 1.7 Weight caching

Private-fork env var `GGML_RPC_CACHE_SCOPE=all|weights|none` gates whether
`set_tensor` for weights goes through `RPC_CMD_SET_TENSOR_HASH` (server-side
content-addressed cache) or a plain `RPC_CMD_SET_TENSOR`. This is orthogonal to
async; it reduces steady-state weight transfer volume but does not change the
serial split model.

### 1.8 Where time goes (qualitative)

Per decode step with N splits across {CUDA0, CUDA1, RPC0}:
- For each split boundary, the scheduler blocks on a cross-backend copy. For
  RPC splits the copy is a full socket round-trip (get from source backend into
  host memory, then set into RPC over TCP), and `ggml_backend_synchronize` on
  the RPC backend is a no-op so it does not protect against overlap.
- `graph_compute` on the RPC split is a third blocking round-trip that holds
  the client thread for the entire remote compute.
- TG is latency-bound: the round-trip latency of each split dominates because
  nothing overlaps. PP is bandwidth-bound on the transient tensor copies into
  RPC (e.g. attention mask tensors noted in `05-performance-bottlenecks.md`).

---

## 2. Async RPC feasibility

### 2.1 What "async" would mean

The ggml backend interface offers three async-shaped hooks:

- `set_tensor_async`, `get_tensor_async`, `set/get_tensor_2d_async` - issue a
  transfer and return immediately.
- `cpy_tensor_async(src_backend, dst_backend, src, dst)` - backend-initiated
  async copy; returns true if handled.
- `event_record` / `event_wait` / `event_synchronize` - backend-internal
  synchronization primitives that the scheduler uses to express dependencies
  without blocking the host thread.
- `graph_compute` is already "async" in name (`ggml_backend_graph_compute_async`
  just calls `iface.graph_compute`); the contract is that the backend may
  enqueue work and return before it finishes, as long as `synchronize` or
  events are used to enforce ordering.

For RPC to participate, it must:
1. Report `caps.async = true`, `caps.events = true`.
2. Implement `event_new/free/record/wait/synchronize` as objects that carry a
   completion signal across the wire.
3. Implement `set_tensor_async`/`get_tensor_async`/`cpy_tensor_async` (or at
   least make `cpy_tensor_async` return false so the scheduler falls back,
   which is the current behavior - but then async events still need to work).
4. Implement `synchronize` to actually wait for outstanding work.

### 2.2 Hard constraints

- **Single socket per endpoint.** `get_socket` returns one shared connection.
  Today the client never issues two commands without receiving the response in
  between, so the single stream is implicitly ordered. Any async scheme must
  either (a) keep strict request/response ordering on that one socket and use
  a client-side worker thread, or (b) open additional connections and tag
  requests with sequence numbers so responses can be demultiplexed.
- **Server is single-threaded per connection.** `rpc_serve_client` handles one
  command at a time. To overlap graph compute with tensor transfer the server
  must either handle multiple in-flight commands on one connection (worker
  pool) or accept multiple connections from the same client.
- **No non-blocking I/O today.** `send_data`/`recv_data` block. Overlap on a
  single thread requires non-blocking sockets or a worker thread.
- **Protocol has no request IDs.** `send_rpc_cmd` sends `cmd` byte + payload
  and reads exactly one response. Async needs correlation.
- `cpy_tensor_async` across a host/RPC boundary would still need a host-side
  staging buffer; the RPC backend buffer is remote, so a cross-backend copy is
  always `get_tensor` (remote->host) + `set_tensor` (host->remote) or vice
  versa. There is no peer-to-peer DMA path except RDMA.

### 2.3 What is feasible

Feasible, in increasing order of effort:

A. **Client-side worker thread, single socket, strict ordering.** Keep the
   wire protocol identical. Run a dedicated I/O thread per endpoint that drains
   a command queue in submission order and posts results to futures. The
   backend's async ops enqueue into this queue and return immediately;
   `synchronize`/`event_wait` block on the future. Overlap is limited because
   the single socket serializes commands server-side, but the client thread can
   issue the next split's input copies while a previous split's graph is
   computing on the server - if and only if the server can handle overlapping
   commands. With the current single-threaded server it cannot, so this option
   alone gives overlap only between local CUDA work and the RPC round-trip for
   other splits, not between RPC compute and RPC transfer.

B. **Server-side worker pool, still one socket, request IDs.** Add a protocol
   version bump: each request carries a 64-bit sequence number and each
   response echoes it. The server dispatches graph_compute (and optionally
   set_tensor) to a thread pool so that a long graph compute does not block a
   short tensor transfer on the same connection. The client I/O thread reads
   responses and matches them by sequence number. This unlocks real overlap of
   compute and transfer for a single client.

C. **Multiple connections per endpoint.** Open N sockets to the same endpoint,
   each with its own server thread. Pin control (graph_compute) to one socket
   and data (set/get/copy) to another, or round-robin. Avoids request-ID
   multiplexing on the wire but doubles connection state and makes ordering
   harder to reason about.

D. **RDMA path.** Already compiled in behind `GGML_RPC_RDMA`. RDMA gives real
   one-sided transfer without round-trips, but only helps the data path
   (set/get/copy), not graph_compute, and requires IB hardware on both ends.
   The M4 Pro side does not have IB, so RDMA is not applicable to this setup.

### 2.4 Conclusion on feasibility

Async RPC is feasible but is **not** a matter of flipping capability bits. The
scheduler's async path assumes a backend that can enqueue work and signal
completion later. RPC's blocking single-socket protocol cannot satisfy that
contract without a client-side worker (option A) at minimum, and cannot deliver
real compute/transfer overlap without a server-side worker pool (option B).
Reporting `caps.async=true` without the underlying machinery would silently
break ordering because `synchronize` is currently a no-op.

---

## 3. Implementation options

Three options, ordered from smallest to largest surface area. Each is described
as a design sketch only - no code is written in this report.

### Option 1: Pipeline-parallel scheduler enablement via fake async (prototype)

**Goal:** unlock the scheduler's existing `parallel=true` path so that splits
on different backends overlap, without touching the wire protocol.

**Shape:**
- Implement `event_new/free/record/wait/synchronize` on the RPC backend as
  **thread-synchronization events**: an event is a `std::atomic<int>` + a
  `std::condition_variable`. `event_record(backend, event)` sets the event
  from the backend's worker; `event_wait(backend, event)` blocks until set;
  `event_synchronize` blocks on the most recent record.
- Implement `synchronize` to actually wait for the RPC graph_compute in flight
  (track a pending future per backend context).
- Report `caps.async = true`, `caps.events = true`.
- Keep `set_tensor_async`/`get_tensor_async`/`cpy_tensor_async` as NULL so the
  scheduler falls back to sync copies (as today), but now wrapped in
  event-based ordering instead of `ggml_backend_synchronize`.

**What it buys:** The scheduler allocates `n_copies = GGML_SCHED_MAX_COPIES`
slots and, crucially, **does not call `ggml_backend_synchronize` between
splits** when events are present. Splits on CUDA0, CUDA1, and RPC0 can execute
back-to-back with only `event_wait`/`event_record` between them. Because
`graph_compute` on RPC still blocks the calling thread, the overlap gained is
mainly between local CUDA splits and the RPC round-trip, and between an RPC
split's compute and the next split's input copies **on the host CPU side**.

**What it does not buy:** True compute/transfer overlap on the RPC socket
itself - the single socket still serializes. The RPC `graph_compute` call still
holds the client thread until the server finishes.

**Risks:**
- The scheduler's `cpy_tensor_async` fallback path calls
  `ggml_backend_synchronize(input_backend)` then `ggml_backend_tensor_copy`.
  With `caps.async=true` but no `cpy_tensor_async`, this path is unchanged, so
  correctness is preserved. But because `synchronize` on RPC was a no-op and
  now must actually wait, any latent assumption that "RPC synchronize is free"
  will surface as new latency.
- Event correctness across the wire is subtle: the event object lives on the
  client; there is no wire event. That is fine for option 1 because the RPC
  graph_compute call is the synchronization point - the event just needs to
  signal "the blocking call returned".
- Must verify that `llama-context.cpp`'s `pipeline_parallel` check (line ~443)
  now passes for RPC. It checks `props.caps.async && props.caps.events` for
  every non-CPU backend. With the change above it will.

**Effort:** Small. One file (`ggml-rpc.cpp`), ~150-300 lines, no protocol
change, no transport change. Backward compatible if gated behind a capability
negotiation bit in `rpc_msg_hello_rsp` (so old servers still work with
`async=false`).

**Verification of correctness:** The event contract in
`ggml-backend.cpp` (`ggml_backend_event_wait`, `event_record`) requires that
`event_wait` on backend B blocks until B has consumed the event-producing
work. Because RPC `graph_compute` is synchronous on the client side, a
client-side event signaled right after `graph_compute` returns is trivially
correct. The risk is in `cpy_tensor_async` paths that are now skipped - but
they are already skipped because RPC leaves them NULL.

---

### Option 2: Client I/O worker + async tensor ops (medium)

**Goal:** let the scheduler issue `set_tensor_async`/`get_tensor_async` and
have them not block the host thread, so that input copies for split N+1 can be
in flight while split N's graph computes locally on a CUDA backend.

**Shape (on top of option 1):**
- Add a per-endpoint worker thread + command queue inside the RPC backend
  context. The worker owns the single socket and runs `send_rpc_cmd`
  request/response pairs in FIFO order.
- `set_tensor_async` enqueues a `RPC_CMD_SET_TENSOR` and returns a future
  (or an internal sequence ID). `get_tensor_async` enqueues a
  `RPC_CMD_GET_TENSOR` and returns a future that the worker fulfills when the
  response arrives.
- `synchronize` flushes the queue and blocks until all outstanding futures
  resolve.
- `cpy_tensor_async` across host/RPC is implemented as `get_tensor_async` into
  a host staging buffer followed by `set_tensor_async`; the event/future
  chains them.
- Events are real futures: `event_record` captures the pending-future state,
  `event_wait` blocks on it.

**What it buys:** Real overlap of (a) local CUDA compute with (b) RPC tensor
transfers, because the host thread is no longer blocked inside `send_rpc_cmd`
during a `set_tensor`. The server is still single-threaded, so RPC commands
for one endpoint still serialize server-side, but they can run concurrently
with local CUDA work on the host.

**What it does not buy:** Overlap of RPC graph compute with RPC tensor
transfer - the single server thread handles one command at a time, so while
the server is in `graph_compute` it cannot service a `set_tensor`. To get
that you need option 3.

**Risks:**
- Ordering: the single socket preserves FIFO order, so as long as the worker
  drains the queue in submission order, request/response matching is trivial
  (no sequence numbers needed). But the caller can now submit a `get_tensor`
  whose response must be read before a later `set_tensor`'s response - the
  worker must not reorder.
- The server's `rpc_serve_client` loop reads one cmd at a time and replies
  before reading the next, so FIFO is preserved by construction. Good.
- Lifetime: tensors referenced by queued async ops must stay alive until the
  worker processes them. Need to reference-count or copy pointers.
- Error propagation: today a failed `send_rpc_cmd` aborts. With a queue, a
  failure must be captured into the future and surfaced at `synchronize` or
  `event_wait`.
- The `get_socket` cache returns a `shared_ptr` shared across buffer ops and
  graph ops; the worker thread must own the socket exclusively, so the
  `socket_t` needs a mutex or the worker must be the sole writer.

**Effort:** Medium. Still one file plus transport is unchanged. ~600-1000
lines. No wire protocol change (still FIFO on one socket). Requires careful
thread-safety review of `ggml_backend_rpc_context` and
`ggml_backend_rpc_buffer_context` (both currently assume single-threaded
access).

---

### Option 3: Server worker pool + request IDs (architectural)

**Goal:** real compute/transfer overlap on the RPC node itself, so that an RPC
`graph_compute` does not block a concurrent `set_tensor` on the same endpoint.

**Shape (on top of option 2):**
- Protocol bump: every request gets a 64-bit sequence number; every response
  echoes it. `RPC_PROTO_MINOR_VERSION` bumped.
- `rpc_serve_client` dispatches `RPC_CMD_GRAPH_COMPUTE` (and optionally
  `RPC_CMD_SET_TENSOR`) to a thread pool. `RPC_CMD_GET_TENSOR` responses are
  written back out of order, matched by sequence number on the client.
- The client I/O thread from option 2 reads a stream of tagged responses and
  fulfills the matching futures.
- Need to decide which commands are safe to run concurrently on the server.
  `graph_compute` on device D is independent of `set_tensor` on device D only
  if they touch different tensors; the server must track in-flight graph
  dependencies. Simplest correct rule: `graph_compute` and `set_tensor` can
  overlap only when they target different devices, or when the set_tensor
  target tensor is not an input to an in-flight graph. This is hard to track
  precisely; a conservative first cut allows `set_tensor` to run concurrently
  with `graph_compute` only on different devices.

**What it buys:** The full theoretical overlap. RPC node can compute a graph
for split N while receiving tensors for split N+1.

**What it does not buy:** Anything if the RPC node has only one device (the
M4 Pro RPC0 has one Metal device), in which case the device serializes anyway
and the server thread pool only helps hide the socket round-trip latency, not
the GPU compute.

**Risks:**
- Protocol versioning: old clients must still work with new servers and vice
  versa. The hello handshake already negotiates a version; a minor bump with a
  capability bit is the clean path.
- Server-side thread safety of `rpc_server` state: `stored_graphs[device]` is
  mutated by `graph_compute` and read by `graph_recompute`. Concurrent
  `graph_compute` for the same device would corrupt `stored_graphs`. Need a
  mutex per device or per stored-graph slot.
- `ggml_backend_graph_compute(backends[device], graph)` calls into the Metal
  backend on the server; Metal compute is itself asynchronous (enqueue +
  commit), so two graphs on the same device can overlap on the server GPU,
  but the second `ggml_backend_graph_compute` will block on the Metal command
  queue unless the backend is reentrant. This needs verification.
- Significant complexity, touches the wire protocol, and must be done as a
  private fork divergence from upstream RPC - any future upstream RPC changes
  will conflict.

**Effort:** Large. ~1500-3000 lines, protocol bump, server threading,
extensive testing. This is the option most likely to be rejected on review if
ever upstreamed, and the option with the highest maintenance burden on the
private fork.

---

## 4. Foreseeable challenges (all options)

1. **Correctness of `synchronize`.** Today `ggml_backend_rpc_synchronize` is a
   no-op and correctness relies on every other call being blocking. Once
   async is introduced, `synchronize` must actually wait. Any code path that
   assumed "RPC synchronize is free" becomes a new blocking point.

2. **Event semantics across the wire.** The scheduler records an event on the
   split backend after a copy and waits on it before reusing the copy slot
   (see `ggml_backend_sched_compute_splits`, the `cur_copy`/`events` logic).
   For RPC these events live client-side; they must correctly model "the RPC
   op I enqueued has completed", which for option 1 is trivial (the blocking
   call already returned) but for option 2+ requires a real future.

3. **Single-socket serialization.** Until option 3, all RPC ops for one
   endpoint share one FIFO stream. The scheduler's async model assumes that
   enqueueing op B after op A means B starts after A completes server-side;
   with a single socket that holds, but it also means no two RPC ops truly
   overlap server-side.

4. **`get_socket` sharing.** The cached `weak_ptr<socket_t>` is shared by
   buffer ops (alloc/free/set/get/copy) and graph ops. A worker thread that
   owns the socket for async ops must coordinate with any sync op that still
   uses the socket directly. Either all ops go through the worker, or a mutex
   serializes direct access vs worker access.

5. **Buffer lifetime.** Async `set_tensor` must keep the source data alive
   until the worker has sent it. Async `get_tensor` must keep the destination
   buffer alive. The scheduler's copy tensors (`tensor_copy(...)`) are
   allocator-managed and live for the graph lifetime, so this is likely fine,
   but user-input tensors (`GGML_TENSOR_FLAG_INPUT`) are copied eagerly today
   precisely to avoid lifetime issues; the async path must preserve that.

6. **MoE expert copy path.** `ggml_backend_sched_compute_splits` has a special
   MoE path that calls `ggml_backend_tensor_get_async` + `synchronize` on the
   ids tensor, then `ggml_backend_tensor_set_async` for selected experts. With
   RPC's async ops NULL, this falls back to sync. If we implement
   `get_tensor_async` (option 2), this path will start using it and must be
   re-verified for correctness.

7. **Telemetry.** The private-fork `rpc_telemetry_*` functions key on
   `ggml_backend_name(backend)`. Async reordering changes the meaning of the
   `copy_ms`/`compute_ms` split logs because the timestamps no longer bracket
   a blocking call. Telemetry will need to record enqueue and completion
   times separately, or be re-scoped.

8. **RDMA interaction.** The RDMA transport path has its own send/recv
   completion model (`rdma_poll`). Any async work must compose with the RDMA
   path, not just TCP. For this setup RDMA is irrelevant (no IB to the Mac),
   but the code path must not break.

9. **`ggml_backend_rpc_device_supports_op` returns true unconditionally.**
   This is already a latent issue but async does not change it.

10. **Reentrancy of the Metal backend on the server.** If option 3 lets two
    `graph_compute` calls target the same Metal device concurrently, the
    Metal backend's command queue serialization must be verified. This is a
    server-side correctness question independent of the wire protocol.

---

## 5. Recommended first implementation

**Option 1: pipeline-parallel enablement via fake async events.**

Rationale:
- It is the smallest change that unblocks the scheduler's existing parallel
  path, which is already written and already works for multi-GPU CUDA. The
  only reason RPC is excluded is the two capability bits and the missing event
  implementations.
- It introduces no wire protocol changes and no new threads on the server.
- It is gateable behind a hello-capability bit so old servers/clients still
  interoperate.
- It gives a measurable signal: if enabling `pipeline_parallel` with RPC
  shows no TG improvement, then the bottleneck is not host-side serialization
  and option 2/3 would not help either - meaning option 1 is also the right
  diagnostic step.
- The private fork already has telemetry (`GGML_RPC_TELEMETRY`) that will
  directly show whether split copy and compute times overlap.

Concrete scope (design only, no code here):
1. Add `ggml_backend_rpc_event` struct: `{ std::atomic<int> signaled; }` (or a
   `std::promise<void>`/`std::future<void>` pair). Implement `event_new`,
   `event_free`, `event_record`, `event_wait`, `event_synchronize` on the RPC
   device interface. `event_record` on RPC just marks the event signaled
   because `graph_compute` has already returned by the time it is called.
2. Change `ggml_backend_rpc_synchronize` from no-op to a real wait on any
   pending in-flight graph (which, for option 1, is none - so it remains a
   no-op, but documented as "no async work is outstanding"). This keeps the
   contract honest for option 2.
3. Flip `caps.async = true` and `caps.events = true` in
   `ggml_backend_rpc_device_get_props`.
4. Leave `set_tensor_async`/`get_tensor_async`/`cpy_tensor_async` NULL so the
   scheduler's copy fallback path is unchanged.
5. Gate the whole thing on a hello capability bit so that a client talking to
   an old server still reports `async=false` (the old server has no event
   support and relies on blocking semantics).

What option 1 will **not** do, and should be stated up front to set
expectations:
- It will not make `graph_compute` non-blocking. The RPC split still holds the
  client thread for its full compute duration.
- It will not overlap RPC compute with RPC transfer (single socket, single
  server thread).
- It will allow local CUDA splits to be issued back-to-back without a
  `ggml_backend_synchronize` between them, and will allow the scheduler to use
  its `n_copies` event slots for RPC, which is the intended first win.

---

## 6. Verification plan

### 6.1 Correctness (all options, before any perf measurement)

1. **Baseline pass.** With async disabled (env var off), confirm the fork still
   produces identical logits/token streams against `master` for the test model
   on the 3-backend setup. This guards against the `synchronize` and event
   changes perturbing the sync path.
2. **Single-split equivalence.** With one backend (no cross-backend copies),
   option 1 should be bit-identical to baseline because the event path is not
   exercised for copies.
3. **Multi-split correctness.** Run prompt-processing and token-generation
   workloads on the full CUDA0+CUDA1+RPC0 setup and compare output tokens
   against baseline. Use `llama-server` with a fixed seed and a pinned prompt.
4. **`ggml_backend_compare_graph_backend` style check.** If feasible, run the
   CPU backend and RPC backend on the same graph and compare outputs, to catch
   event-ordering bugs that only manifest with `n_copies > 1`.
5. **MoE models.** If any MoE model is available, run it - the
   `cpy_tensor_async`/expert-copy path in the scheduler is the most fragile
   part and must not regress. With option 1 those ops stay sync, so this is a
   "must not break" check, not a "new behavior" check.
6. **Stress.** Long-running decode (1000+ tokens) with the 3-backend split to
   surface event-slot reuse bugs (`cur_copy` rotates through
   `GGML_SCHED_MAX_COPIES`).

### 6.2 Performance signal (option 1)

1. **Telemetry diff.** Run with `GGML_RPC_TELEMETRY=1` before and after.
   Primary metric: per-split `compute_ms` and `copy_ms` for the RPC backend.
   With option 1, the copy and compute times themselves should be unchanged,
   but the **wall-clock** decode time should drop if splits now overlap.
2. **Wall-clock TG.** Measure tokens/sec for single-token decode with the
   3-backend split, option 1 on vs off. Expectation: modest improvement if the
   host was serializing; no regression if it was not.
3. **Wall-clock PP.** Measure tokens/sec for large-batch prompt processing.
   Expectation: smaller relative improvement because PP is bandwidth-bound and
   option 1 does not change transfer volume.
4. **Nsight Systems / Instruments.** Capture a timeline of the host process
   during decode. Look for whether CUDA0/CUDA1 splits now overlap with the RPC
   round-trip. The RPC round-trip should appear as a gap in CPU activity
   rather than a barrier.
5. **Isolation.** Disable RPC0 and run CUDA0+CUDA1 only to confirm that
   option 1 does not regress the pure-CUDA multi-GPU path (it should not touch
   it, but the `caps.async`/`events` check is shared).

### 6.3 Decision gate after option 1

If option 1 shows no TG improvement, the bottleneck is elsewhere (likely the
single-socket serialization or the RPC `graph_compute` blocking call). At that
point option 2 (client I/O worker) is the next step, and its verification plan
is:

- Repeat 6.1 (correctness must still hold with async tensor ops).
- Add a check that `set_tensor_async` actually returns before the server has
  acked (use telemetry timestamps: enqueue time vs completion time).
- Measure whether RPC `copy_ms` now overlaps with local CUDA `compute_ms` in
  the telemetry summary.
- Option 3 (server worker pool) is only justified if option 2 shows that RPC
  compute and RPC transfer are both significant and serial on the server.

### 6.4 Regression guards

- Keep `GGML_RPC_TELEMETRY` capable of running with async off (env-gated) so
  before/after comparisons are apples-to-apples.
- Add an env var `GGML_RPC_ASYNC=0|1` (default off in early iterations) so the
  capability flip is runtime-switchable without recompiling, to bisect any
  correctness regression to the async change.
- The hello-capability negotiation must fall back cleanly: if the server does
  not advertise event support, the client must report `caps.async=false` and
  the scheduler takes the existing sync path. This guarantees old servers keep
  working.
