# Upstream RPC Merge Report

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

## Outcome

The upstream RPC changes are beneficial to this fork and have been integrated locally. The merged RPC targets build successfully. Runtime validation with a matched client and server is still required before the merge is finalized.

## Adopted From Upstream

- One shared `rpc_dispatcher` and worker thread per endpoint.
- Ordered synchronous and asynchronous command submission.
- Asynchronous graph compute and graph recompute.
- Asynchronous tensor set and get entry points.
- Real queue-backed events and synchronization.
- RPC device capabilities report async and event support.
- Automatic transport negotiation with TCP fallback.
- Linux RoCE/InfiniBand support through `libibverbs`.
- Apple silicon RDMA over Thunderbolt through `librdma`.

## Retained From The Private Fork

- `GGML_RPC_CACHE_SCOPE=all|weights|none`.
- `GGML_RPC_TELEMETRY=1`.
- `LLAMA_RPC_SPLIT_SCALE`.
- `LLAMA_OUTPUT_LAYER_DEVICE`.
- Historical measurements and operational guidance.

Cache scope applies to synchronous and asynchronous tensor uploads. A cache hit avoids the full upload. A miss sends:

```text
rpc_tensor | offset | cache_write | data
```

## Removed

`GGML_RPC_PIPELINE` and its fake no-op events were removed. They are superseded by upstream's real dispatcher and event implementation. Old benchmark results involving that switch remain in the notes as explicitly labeled history.

## Protocol Compatibility

This fork uses RPC protocol `7.0.0`. The private `cache_write` field changes the `RPC_CMD_SET_TENSOR` wire format, so a major version boundary is required.

Always run matching client and server binaries from this fork. A protocol 7 client rejects an upstream protocol 6 server, and an older client rejects a protocol 7 server during `HELLO`.

## Telemetry Semantics

Graph submission is asynchronous. Client graph telemetry therefore reports:

- `serialize_ms`: graph serialization time.
- `enqueue_ms`: time to place the command on the dispatcher queue.

`enqueue_ms` is not network or compute duration. Use these measurements for the critical path:

- RPC server `compute_ms`.
- dependent client `get_tensor wait_ms`.
- scheduler split copy and compute timing.
- end-to-end prompt-processing and token-generation speed.

Cache hash checks are still synchronous because their result decides whether the tensor payload must be sent. On a cache miss, asynchronous tensor upload queues the payload.

## Transport Notes

TCP is always available. RDMA upgrades automatically when both peers advertise compatible support. Use `GGML_RPC_NO_RDMA=1` on either peer to force TCP.

For Apple RDMA over Thunderbolt:

- use Apple silicon with supported Thunderbolt hardware;
- use macOS 26.2 or later;
- enable RDMA once from Recovery with `rdma_ctl enable`;
- connect using the peer address on the Thunderbolt link.

## Verification Status

Completed:

- pre-merge RPC-enabled baseline build;
- merge conflict resolution;
- no remaining conflict markers;
- post-merge build of `ggml-rpc-server` and `llama-server`.

Still required:

- start the protocol 7 RPC server;
- confirm the client discovers the remote device;
- run a short inference smoke test;
- confirm cache scope behavior;
- capture one telemetry run;
- compare TCP and RDMA only if the current hardware supports RDMA.
