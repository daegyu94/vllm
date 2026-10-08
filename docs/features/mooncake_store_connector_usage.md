# MooncakeStoreConnector Usage Guide

MooncakeStoreConnector is a KV cache connector that uses [MooncakeDistributedStore](https://github.com/kvcache-ai/Mooncake) as a shared KV cache pool. Unlike `MooncakeConnector` which does direct point-to-point KV transfer between prefiller and decoder, MooncakeStoreConnector enables KV cache offloading to an external distributed store, supporting:

- **CPU/disk offloading**: Extend effective KV cache capacity by offloading to CPU memory or disk via Mooncake's transfer engine.
- **Prefix caching across instances**: Hash-based deduplication allows multiple vLLM instances to share cached KV blocks through the store.
- **Single-node and multi-node deployment**: Works both as a standalone KV cache extension and in disaggregated prefill-decode setups.

## Prerequisites

### Install Mooncake

Install mooncake through pip:

```bash
uv pip install mooncake-transfer-engine
```

Refer to the [Mooncake official repository](https://github.com/kvcache-ai/Mooncake) for more installation instructions and building from source.

### Start the Mooncake Master Server

The Mooncake master manages metadata and coordinates the distributed store. Start it before launching vLLM:

```bash
mooncake_master --port 50051
```

Default ports:

- RPC: 50051

Multiple vLLM instances can share the same master server.

### Configure Mooncake

Create a JSON configuration file (e.g., `mooncake_config.json`):

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
```

- `mode`: Topology selection. `"embedded"` (default, PR-40900 baseline) has each
  vLLM rank contribute `global_segment_size` to the pool in-process.
  `"standalone-store"` makes ranks pure requesters — an external
  `mooncake_client` process owns the CPU pool and (optionally) the SSD tier.
- `protocol`: Use `"rdma"` for best performance. `"tcp"` works as a fallback.
- `global_segment_size`: CPU memory contributed to the distributed pool (per
  GPU). Must be `> 0` in `embedded` mode and `0` in `standalone-store` mode.
- `local_buffer_size`: Private buffer for this node's own operations (per GPU).
- `enable_offload`: When `true`, vLLM allocates a DirectIO staging buffer so
  large prefills do not exceed the owner's SSD-write budget. Set this together
  with the matching `--enable_offload=true` flag on `mooncake_master` and on
  the external `mooncake_client` (if any).
- `tenant_id`: Optional Mooncake tenant namespace. Producers and consumers
  that should share store data must use the same tenant id. Default:
  `"default"`.
- `enable_dfs`: Initialize the client's file-storage layer and request a DFS
  replica for each KV put. Defaults to `false` and is separate from
  `enable_offload`, which controls the existing owner SSD tier. Requires a
  DFS-enabled Mooncake build and the `default` tenant.

Set the config path via environment variable:

```bash
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```

## Usage

### Distributed File Storage (DFS)

Set `"enable_dfs": true` in the Mooncake JSON configuration. Configure the Master and every reader/writer with the same Mooncake DFS root, backend, shard layout, and alignment, following the [Mooncake DFS deployment guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/deployment/mooncake-store-deployment-guide.md). The client is initialized with `enable_ssd_offload=True`. Each put requests one memory replica and one DFS replica by default. A Mooncake build without `ReplicateConfig.dfs_replica_num` is rejected before initializing the Store.

Memory replication can be increased through `kv_connector_extra_config`:

```json
{
  "kv_connector": "MooncakeStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "replica_num": 2,
    "dfs_replica_num": 1
  }
}
```

Mooncake requires `replica_num >= 1` and `dfs_replica_num == 1` for DFS. DFS GET batches use the client's `local_buffer_size` as their staging limit, also respecting the existing SSD limit when `enable_offload` is enabled. Individual keys that exceed the usable staging budget are reported as load failures instead of being submitted as an oversized batch. Concurrent reads share the native buffer and can still exhaust it.

To verify DFS I/O, save KV, drain pending puts, then remove only the memory replicas. Check that the remaining DFS descriptors restore identical data and that the storage backend reports actual reads. A memory cache hit or a successful rollout alone does not verify the DFS path. Use a dedicated Master and DFS root for this verification because a connector reset clears the Store's tenant key space.

### Single-Node KV Cache Offloading

Use MooncakeStoreConnector to offload KV cache to CPU memory, extending the effective cache size:

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

### Disaggregated Prefill-Decode (XpYd)

In disaggregated prefill-decode mode, use `MultiConnector` to combine `MooncakeConnector` (point-to-point KV transfer) with `MooncakeStoreConnector` (shared KV cache pool). This enables both direct P2P transfer between prefiller and decoder, and cross-instance prefix cache sharing via the distributed store.
**Prefiller Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50052 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8100 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_producer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both"
                }
            ]
        }
    }'
```

**Decoder Node:**

```bash
MOONCAKE_CONFIG_PATH=mooncake_config.json \
VLLM_MOONCAKE_BOOTSTRAP_PORT=50053 \
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --port 8200 \
    --kv-transfer-config '{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "connectors": [
                {
                    "kv_connector": "MooncakeConnector",
                    "kv_role": "kv_consumer"
                },
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_consumer"
                }
            ]
        }
    }'
```

To also offload newly completed decode KV blocks, add the following extra
configuration to the decoder's `MooncakeStoreConnector` entry.

When decode processing starts, the consumer checks the block-aligned prompt
prefix and fills any blocks missing from the Store. Subsequent saves append
newly completed decode blocks. This keeps a complete, reusable prefix in the
Store. It also covers prompt KV delivered directly by `MooncakeConnector`.

```json
{
    "kv_connector_extra_config": {
        "save_decode_cache": true
    }
}
```

#### Sharing one Store across multiple Prefill TP sizes

Heterogeneous-TP sharing normally uses a fixed `store_tp_size`. When several
prefillers use different TP sizes, opt in to a common Store TP derived from
their least common multiple:

```json
{
    "kv_connector_extra_config": {
        "enable_store_tp_lcm": true,
        "prefill_tp_sizes": [4, 2]
    }
}
```

Every prefiller and decoder that shares these entries must use the same list.
The example selects Store TP 4: a TP4 endpoint maps each rank to one Store
shard, while TP2 endpoints map each rank to two Store shards. Runtime TP sizes
remain unchanged. A decoder configured with `"save_decode_cache": true` uses
the same Store TP for decode KV from every prefiller.

The list may contain positive integer TP sizes. Sharing requires a Store TP that
is at least the local TP and divisible by it, an LBHNC or LBNHC local KV cache,
and the existing topology and KV-head constraints. The Store namespace includes
the attention backend's selected layout. Different layouts use separate Store
entries. Malformed lists and unsupported endpoints use an isolated rank-local
key layout. When
`enable_store_tp_lcm` is absent or false, `prefill_tp_sizes` has no effect and
the existing `store_tp_size` behavior is unchanged.

**Proxy:**

A disaggregation proxy routes requests between prefiller and decoder nodes.
When `MooncakeConnector` is also used for direct P2P transfer, refer to its
[usage guide](mooncake_connector_usage.md) for proxy setup details.

### Disk Offloading

Disk offloading is most commonly run in `standalone-store` mode: an external
`mooncake_client` process owns the CPU pool and the SSD tier, and each vLLM
rank is a pure requester. This avoids per-rank duplication of the SSD pool
and keeps DirectIO budget tracking on a single process.

Three things need to be aligned for end-to-end disk offloading:

1. **`mooncake_master`** is started with `--enable_offload=true`.
2. **`mooncake_client`** (the owner) is started with `--enable_offload=true`
   plus an SSD path via `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`.
3. **vLLM-side** sets `"enable_offload": true` in the JSON config file (this is
   read by the connector and is **not** an environment variable).

Example `mooncake_config.json` for the vLLM side:

```json
{
  "mode": "standalone-store",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": 0,
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": true
}
```

Steer this rank to the local owner segment with:

```bash
export MOONCAKE_PREFERRED_SEGMENT=127.0.0.1:50053
```

The owner's SSD directory, on-disk eviction policy, and the DirectIO staging
buffer size are controlled on the `mooncake_client` side via the standard
Mooncake environment variables (`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`,
`MOONCAKE_BUCKET_EVICTION_POLICY`, `MOONCAKE_USE_URING`,
`MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`,
`MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`, etc.). Those are independent of
the vLLM JSON config.

### Tenant Isolation

Set `tenant_id` in the Mooncake JSON config when different vLLM deployments should use separate Mooncake tenant namespaces:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "global_segment_size": "80GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false,
  "tenant_id": "tenant-a"
}
```

Strict isolation requires a Mooncake master started with `--enable_multi_tenants=true` and a tenant quota policy that registers each tenant. Non-default `tenant_id` also requires a Mooncake version whose `MooncakeDistributedStore.setup()` accepts the `tenant_id` parameter. In `standalone-store` mode, start the external `mooncake_client` with the matching tenant id because that process owns the real store client.

## Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `MOONCAKE_CONFIG_PATH` | Path to Mooncake JSON config file | (required) |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` | Bootstrap port for MooncakeConnector P2P transfer (disagg mode only) | 8998 |
| `MOONCAKE_PREFERRED_SEGMENT` | Pin this rank's replicas to a specific owner segment (`host:port`); used in `standalone-store` mode | — |
| `MOONCAKE_REQUESTER_LOCAL_HOSTNAME` | Override the hostname the vLLM rank registers with Mooncake as a requester. Defaults to the rank's resolved IP. | — |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | When `1`, logs a per-batch tier summary (memory vs disk hits) for observability | disabled |
| `VLLM_MOONCAKE_DISK_STAGING_USABLE_RATIO` | Fraction of the owner's DirectIO staging buffer that the requester will fill in a single `batch_get_into_multi_buffers` call. Lower → more conservative pre-split, more round trips. | 0.9 |

## KV Transfer Config

### KV Role Options

- **kv_producer**: For instances that store KV caches to the pool.
- **kv_consumer**: For instances that load KV caches from the pool.
- **kv_both**: The instance both stores and loads KV caches. Use this for single-node CPU offloading or prefiller instances.

### kv_connector_extra_config

- `load_async` (bool): Enable asynchronous loading for better compute-I/O overlap. Default: `true`.
- `lookup_async` (bool): Run the external prefix-cache lookup on a background thread so it never blocks the scheduler step. The request is held until the in-flight lookup completes, then resumed on a later step. Default: `false`.
- `lookup_rpc_port` (int): Custom port for the ZMQ lookup RPC socket. Default: `0`.
- `cache_prefix` (str): Namespace prepended to every store key. Use distinct prefixes for deployments with different weights, and the same prefix only for instances sharing one policy. A nonempty prefix also scopes connector resets to that namespace. `@` is reserved as the key delimiter and cannot appear in a prefix. Default: `""` (unprefixed keys and legacy tenant-wide reset).
- `save_decode_cache` (bool): Enable offloading decode tokens' KV cache. A `kv_consumer` does not save during prefill; when decode starts, it fills any missing block-aligned prompt prefix before appending completed decode blocks. Default: `false`.
- `store_tp_size` (int): Common Store TP for endpoints with different local TP sizes. It supports LBHNC and LBNHC local KV caches, with `store_tp_size >= local_tp_size` and `store_tp_size % local_tp_size == 0`. The current topology is one full-attention cache group, PCP/DCP disabled, and cross-layer blocks disabled. For GQA and MHA, the total KV-head count must be divisible by `store_tp_size`. Store shards contain fixed global KV-head ranges in the local layout. Shared endpoints use the same KV cache layout, pipeline-parallel size, and Store TP. The Store namespace includes the layout and PP size. Unsupported configurations use a topology-specific rank-local namespace.

LBHNC/HND is strongly recommended for TP-sharded Store when supported. LBNHC/NHD
creates many transfer segments and may significantly reduce PUT/GET performance.

Before resetting a shared namespace, pause admission on every replica using that prefix. Prepare every TP/DP worker with `prepare_kv_cache_reset`: drain local transfers, synchronize CUDA work, then invalidate the matching client-local cache. Async clients expose `prepare_kv_cache_reset()` across all managed engine cores. Only after every participant acknowledges preparation may any scheduler reset the shared namespace; resume only after all resets succeed.

`reset_prefix_cache(reset_running_requests=True, reset_connector=True)` keeps preempted requests in the waiting queue for recomputation. Inactive waiting requests do not block reset, but in-flight batches/tokens, pending remote KV receives, deferred block frees, or running requests without the preemption flag do. Workers drain before request blocks are released. A failed local KV reset never deletes shared keys; callers must complete pending work before retrying.

A connector reset requires matching Mooncake Master/SDK support for `remove_by_regex_strict`, `remove_all_strict`, and `invalidate_local_cache`. Strict deletion checks matching metadata and active processing/task entries, including keys omitted from readable-key listings. Partial removal returns an error; a verified empty namespace may return `0` on an idempotent retry. Missing SDK support fails closed and does not fall back to best-effort or tenant-wide deletion. The native RealClient path implements these APIs; unsupported IPC/dummy clients return an error.

On a split GET error or exception, the receiver conservatively invalidates every planned block of the request, including successfully loaded and unattempted blocks. This preserves the original prefix boundary under TP request rotation, at the cost of extra recomputation. This does not implement I/O cancellation/deadlines or coordinate writers outside the caller's participant set.

For example, with prefill TP 4, decode TP 2, and eight KV heads, set
`store_tp_size` to 4 on both instances. Each decode rank reads and writes two
of the four Store shards.

MQA with one total KV head uses a replicated-head layout. For the supported
prefill TP 4 to decode TP 2 case, every rank uses the
same rank-0 key namespace. The four prefill replicas stripe block PUTs so each
object is stored once, while both decode ranks GET every block into their local
KV replica. `store_tp_size` does not appear in MQA keys, so identical MQA
objects written at different store TP sizes share the same pool entry when PP
sizes match.

Tensor-parallel collectives and low-precision arithmetic are not bitwise
invariant across TP sizes, so heterogeneous-TP reuse does not guarantee the
same greedy output as recomputing the prefix at the decode TP size.

## Notes

### Reproducible Block Hashes Across Processes

The `MooncakeStoreConnector` relies on consistent block hashes across all vLLM processes sharing the distributed store. Block hashes chain from `NONE_HASH`, which is derived from a fixed default seed, so identical prompts produce identical block hashes across processes by default — enabling cross-process prefix cache hits without extra configuration.

The exception is the non-cryptographic `xxhash`/`xxhash_cbor` values of `--prefix-caching-hash-algo`, which seed `NONE_HASH` randomly per process; sharing a store with those requires `PYTHONHASHSEED`.

To use a custom shared seed, set the same `PYTHONHASHSEED` on every instance that shares the store (DP ranks, separate prefiller/decoder nodes, and any other vLLM process pointed at the same Mooncake store):

```bash
PYTHONHASHSEED=<shared-value> vllm serve ...
```
