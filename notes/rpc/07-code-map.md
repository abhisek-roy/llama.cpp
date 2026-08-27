# Code Map

Reviewed against upstream `192067b72` and the private-fork merge resolution of 2026-08-27.

This page maps architecture concepts to source files.

## User Options

- `common/arg.cpp`
  - `add_rpc_devices()`: parses `--rpc` server list and registers RPC devices.
  - `--device`: restricts device list.
  - `--n-gpu-layers`: controls how many layers can be offloaded.
  - `--split-mode`: selects none, layer, row, or tensor mode.
  - `--tensor-split`: sets manual split proportions.
  - `--main-gpu`: selects the single device in none mode.
  - `--kv-offload` and `--no-kv-offload`: controls KV offload.

- `common/common.cpp`
  - `common_model_params_to_llama()`: copies common CLI params into `llama_model_params`.

## RPC Server

- `tools/rpc/rpc-server.cpp`
  - `main()`: parses server flags, loads backends, selects devices, starts RPC server.
  - `get_devices()`: selects exposed devices. If none are requested, exposes accelerators first, CPU fallback otherwise.

- `ggml/include/ggml-rpc.h`
  - public RPC backend API.
  - protocol version constants. This fork is `7.0.0` because its `cache_write` byte changes the `RPC_CMD_SET_TENSOR` wire layout.

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
  - `rpc_cmd`: binary protocol commands.
  - `rpc_tensor`: serialized tensor metadata. The 4-byte padding field is now `int32_t use_count`. Wire size unchanged.
  - `rpc_dispatcher`: one ordered command queue and worker thread per endpoint.
  - `rpc_dispatcher::send_async()`: queues tensor or graph work without waiting for socket completion.
  - `rpc_dispatcher::synchronize()` and event methods: queue-backed completion fences.
  - `ggml_backend_rpc_add_server()`: creates a backend registry for one RPC endpoint.
  - `ggml_backend_rpc_init()`: creates a backend instance for a remote device.
  - `ggml_backend_rpc_buffer_type()`: creates RPC buffer type for a remote device.
  - `ggml_backend_rpc_buffer_set_tensor()`: synchronously sends tensor bytes through the dispatcher.
  - `ggml_backend_rpc_set_tensor_async()` and `ggml_backend_rpc_get_tensor_async()`: queue asynchronous tensor traffic.
  - `ggml_backend_rpc_graph_compute()`: queues graph compute or graph recompute.
  - `add_tensor()`: walks the split's tensors and packs each one's global use count from the cgraph `visited_hash_set`.
  - `rpc_server::graph_compute()`: reconstructs and executes remote graph. Restores the received use counts into the remote graph.
  - `rpc_server::graph_recompute()`: reruns stored graph.

- `ggml/src/ggml-rpc/transport.cpp`
  - socket transport and capability negotiation.
  - TCP fallback and Linux RoCE/InfiniBand support through `libibverbs`.
  - `GGML_RPC_NO_RDMA`: runtime opt-out from an available RDMA transport.

- `ggml/src/ggml-rpc/transport-apple.cpp`
  - Apple silicon RDMA-over-Thunderbolt implementation through `librdma`.

- `ggml/src/ggml-rpc/CMakeLists.txt`
  - detects Linux or Apple RDMA libraries and enables `GGML_RPC_RDMA` when available.

## Device And Weight Placement

- `src/llama.cpp`
  - `llama_supports_rpc()`: checks whether RPC backend registry exists.
  - `llama_prepare_model_devices()`: builds the model device list.
  - default path inserts RPC devices before local GPUs.

- `include/llama.h`
  - `llama_split_mode`: public split mode enum.
  - `llama_model_params`: devices, split mode, tensor split, main GPU, GPU layer count.

- `src/llama-model.cpp`
  - `llama_model_base::load_tensors()`: computes layer-device placement and allocates weight buffers.
  - `make_cpu_buft_list()`: builds CPU and host buffer candidates.
  - `make_gpu_buft_list()`: builds device buffer candidates.
  - `llama_model::n_gpu_layers()`: negative values mean all layers plus output layer.
  - `llama_model::dev_layer()`: returns device assigned to one layer.
  - `llama_meta_device_get_split_state()`: tensor-parallel split metadata.

- `src/llama-model-loader.cpp`
  - `llama_model_loader::create_tensor()`: classifies tensors and chooses a compatible buffer type.

## Inference And Scheduling

- `src/llama-context.cpp`
  - `llama_context::decode()`: main decode loop over ubatches.
  - `llama_context::process_ubatch()`: applies memory, builds or reuses graph, sets inputs, computes graph.
  - `llama_context::graph_compute()`: calls scheduler compute.
  - `llama_context::sched_reserve()`: reserves prompt-processing and token-generation graph shapes.
  - `llama_context::graph_get_cb()`: guides some layer backend placement for norm and final layer ops.

- `src/llama-graph.cpp`
  - graph construction helpers and reusable graph result structures.

- `ggml/src/ggml-backend.cpp`
  - `ggml_backend_sched_split_graph()`: assigns nodes to backends and creates backend splits.
  - `ggml_backend_sched_compute_splits()`: copies split inputs and computes each split.
  - `ggml_backend_sched_backend_id_from_cur()`: uses existing buffers and weight ownership to choose a backend.

- `ggml/src/ggml.c`
  - `ggml_visit_parents_graph()`: builds the cgraph `visited_hash_set` and `use_counts` at graph build time.
  - `ggml_graph_view()`: scheduler splits share the full graph's `use_counts` and hash set, so RPC splits see global counts.

- `ggml/src/ggml-impl.h`
  - `ggml_node_get_use_count()`: reads a node's use count from the cgraph hash set.
  - `ggml_node_has_n_uses()`: fusion check, used by exactly N nodes so it can be fused.

- `ggml/src/ggml-backend-meta.cpp`
  - MoE AllReduce delay logic, the main in-tree consumer of use counts.
  - copies full-graph use counts into per-device subgraphs when splitting.

## KV Cache

- `src/llama-kv-cache.cpp`
  - KV cache allocation uses CPU buffer by default.
  - With KV offload, KV buffer type follows `model.dev_layer(il)`.
  - `get_k()` and `get_v()` return views into cache tensors.
  - `cpy_k()` and `cpy_v()` write current K/V into cache through graph ops.

## Existing Docs

- `tools/rpc/README.md`: official RPC usage and warnings.
- `docs/multi-gpu.md`: split modes and multi-GPU controls.
- `docs/development/token_generation_performance_tips.md`: general token generation troubleshooting.
