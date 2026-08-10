# Code Map

Traceability source commit: `b89f44654161`

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
  - protocol version constants.

- `ggml/src/ggml-rpc/ggml-rpc.cpp`
  - `rpc_cmd`: binary protocol commands.
  - `rpc_tensor`: serialized tensor metadata.
  - `ggml_backend_rpc_add_server()`: creates a backend registry for one RPC endpoint.
  - `ggml_backend_rpc_init()`: creates a backend instance for a remote device.
  - `ggml_backend_rpc_buffer_type()`: creates RPC buffer type for a remote device.
  - `ggml_backend_rpc_buffer_set_tensor()`: sends tensor bytes to the server.
  - `ggml_backend_rpc_graph_compute()`: sends graph compute or graph recompute command.
  - `rpc_server::graph_compute()`: reconstructs and executes remote graph.
  - `rpc_server::graph_recompute()`: reruns stored graph.

- `ggml/src/ggml-rpc/transport.cpp`
  - `socket_t::impl::send_data()`: blocking send loop.
  - `socket_t::impl::recv_data()`: blocking recv loop.
  - RDMA path if built with `GGML_RPC_RDMA`.

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
