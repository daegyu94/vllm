# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay identical growing prefixes through the actual sender and native DFS.

No LLM generation occurs here. Source buffers are CUDA-resident, with one
logical 448 KiB KV block scattered across 56 layer/KV segments. Every policy
uses the same token hashes and changed payload bytes to check invalidation.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True)
    parser.add_argument("--dedicated-master-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    if (
        not args.dedicated_master_confirmed
        or os.environ.get("MOONCAKE_DFS_FS_ADAPTER") != "hf3fs"
    ):
        parser.error("requires a dedicated real hf3fs master")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
        ChunkedTokenDatabase,
        KeyMetadata,
        ReqMeta,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "offload").mkdir()
    slices, blocks, slice_bytes = 56, 32, 8192
    block_bytes = slices * slice_bytes
    sources = [
        torch.empty(blocks * slice_bytes, dtype=torch.uint8, device="cuda:0")
        for _ in range(slices)
    ]
    destination = torch.empty(block_bytes, dtype=torch.uint8, device="cuda:0")
    store = MooncakeDistributedStore()
    assert (
        store.setup(
            "127.0.0.1:53401",
            "P2PHANDSHAKE",
            0,
            134217728,
            "tcp",
            "",
            args.master,
            enable_ssd_offload=True,
            ssd_offload_path=str(args.output / "offload"),
        )
        == 0
    )
    for buf in sources + [destination]:
        assert store.register_buffer(buf.data_ptr(), buf.numel()) == 0
    config = ReplicateConfig()
    config.replica_num, config.dfs_replica_num = 1, 1
    db = ChunkedTokenDatabase(KeyMetadata("replay", 0, 0, 0, 0), block_size=16)
    db.set_kv_caches_base_addr([buf.data_ptr() for buf in sources])
    db.set_block_len([slice_bytes] * slices)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=None)
    coord = worker.MooncakeStoreCoordinator(
        [KVCacheGroupSpec(["layer0"], spec)],
        scheduler_block_size=16,
        hash_block_size=16,
    )
    events = []
    sender = worker.KVCacheStoreSendingThread(
        store,
        coord,
        [db],
        16,
        0,
        1,
        "kv_producer",
        threading.Event(),
        replicate_config=config,
        save_request_deltas=args.delta,
        record_operation=lambda operation,
        duration_seconds,
        num_keys,
        **kw: events.append(
            dict(op=operation, duration_seconds=duration_seconds, keys=num_keys, **kw)
        ),
    )
    runs = []
    try:
        for epoch in range(args.epochs):
            mounted = store.allocate_and_mount_segment(268435456)
            assert mounted["ret"] == 0
            expected = {}
            wall_total = cpu_total = 0
            for request in range(2):
                # The two requests share the first 16 blocks, then diverge.
                identifiers = [i if i < 16 else i + 16 * request for i in range(blocks)]
                hashes = [
                    hashlib.sha256(f"prefix-{i}".encode()).digest() for i in identifiers
                ]
                values = [(epoch * 53 + i) % 251 + 1 for i in identifiers]
                for buf in sources:
                    for block, value in enumerate(values):
                        buf[block * slice_bytes : (block + 1) * slice_bytes].fill_(
                            value
                        )
                torch.accelerator.synchronize()
                request_id = f"request-{request}"
                for count in range(8, blocks + 1):
                    req = ReqMeta(
                        req_id=request_id,
                        token_len_chunk=count * 16,
                        block_ids=(list(range(blocks)),),
                        block_hashes=hashes[:count],
                        can_save=True,
                    )
                    sender.add_stored_request(request_id)
                    sender.request_queue.put(req)
                    start, cpu = time.perf_counter_ns(), time.thread_time_ns()
                    sender._handle_request(req)
                    wall_total += time.perf_counter_ns() - start
                    cpu_total += time.thread_time_ns() - cpu
                for _, end, key in db.process_tokens(blocks * 16, hashes):
                    expected[key.to_string()] = values[end // 16 - 1]
                if request == 0:
                    sender.delete_finished_stored_request(request_id)
            sender.request_queue.join()
            assert all(value == 0 for value in sender.stored_requests.values())
            assert store.unmount_and_free_segment(mounted["segment_ids"], 0) == 0
            # Full DFS restoration checks actual bytes, not success codes alone.
            for key, value in expected.items():
                assert not any(
                    d.is_memory_replica() for d in store.get_replica_desc(key)
                )
                destination.zero_()
                torch.accelerator.synchronize()
                result = store.batch_get_into_multi_buffers(
                    [key], [[destination.data_ptr()]], [[block_bytes]]
                )
                assert result == [block_bytes], result
                torch.accelerator.synchronize()
                assert bool(torch.all(destination == value)), key
            known = sum(len(v) for v in sender._saved_request_keys.values())
            sender.clear_saved_request_keys()
            assert not sender._saved_request_keys
            assert store.remove_all(force=True) >= 0
            assert store.batch_is_exist(list(expected)) == [0] * len(expected)
            runs.append(
                dict(
                    epoch=epoch,
                    sender_wall_ns=wall_total,
                    sender_thread_cpu_ns=cpu_total,
                    published_keys=len(expected),
                    remembered_keys_before_reset=known,
                    stale_hits=0,
                )
            )
    finally:
        store.close()
    result = dict(
        level="real vLLM sender + Mooncake/3FS synthetic replay; not Agent RL training",
        delta=args.delta,
        sender_source=worker.__file__,
        lifecycle=runs,
        source_sha256=hashlib.sha256(Path(worker.__file__).read_bytes()).hexdigest(),
        native_module=__import__("mooncake.store", fromlist=["store"]).__file__,
        logical_put_bytes=len(expected) * block_bytes * args.epochs,
        logical_dfs_read_bytes=len(expected) * block_bytes * args.epochs,
        sender_wall_ns=sum(r["sender_wall_ns"] for r in runs),
        sender_thread_cpu_ns=sum(r["sender_thread_cpu_ns"] for r in runs),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    for operation in ["save_exists", "save_put"]:
        selected = [e for e in events if e["op"] == operation]
        result[operation] = dict(
            calls=len(selected),
            keys=sum(e["keys"] for e in selected),
            seconds=sum(e["duration_seconds"] for e in selected),
        )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
