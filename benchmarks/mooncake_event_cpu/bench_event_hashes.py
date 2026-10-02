# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only worker method microbenchmark; all store operations are stubs."""

import ast
import gc
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path
from types import MethodType

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store import worker

root = Path(worker.__file__).parents[7]
relative = str(Path(worker.__file__).relative_to(root))
spec = importlib.util.spec_from_file_location(
    "worker_helpers", root / "tests/v1/kv_connector/unit/test_mooncake_store_worker.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
_make_store_req = helpers._make_store_req
_make_store_sending_thread = helpers._make_store_sending_thread
baseline = subprocess.check_output(
    ["git", "-C", str(root), "show", "integration/pinned-3fs-baseline:" + relative],
    text=True,
)
functions = {}
for variant, source in [
    ("baseline", baseline),
    (
        "patched",
        subprocess.check_output(
            ["git", "-C", str(root), "show", "perf/mooncake-event-hashes:" + relative],
            text=True,
        ),
    ),
]:
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KVCacheStoreSendingThread"
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_handle_request"
    )
    namespace = dict(worker.__dict__)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), variant, "exec"), namespace)
    functions[variant] = namespace["_handle_request"]


class Store:
    def __init__(self, hit):
        self.hit = hit
        self.last = None

    def batch_is_exist(self, keys):
        return [self.hit] * len(keys)

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, replica):
        self.last = (keys, addrs, sizes)
        return [0] * len(keys)


hashes = [i.to_bytes(32, "little") for i in range(64)]
request = worker.ReqMeta(
    req_id="bench",
    token_len_chunk=16 * len(hashes),
    block_ids=(list(range(len(hashes))),),
    block_hashes=hashes,
    can_save=True,
)
results = []
for hit in [0, 1]:
    for enabled in [False, True]:
        outputs = {}
        for variant, fn in functions.items():
            store = Store(hit)
            thread = _make_store_sending_thread(store)
            thread.request_queue.task_done = lambda: None
            thread.enable_kv_event = enabled
            thread._handle_request = MethodType(fn, thread)

            def iteration(thread=thread):
                thread.add_stored_request("bench")
                thread._handle_request(request)
                thread.kv_events.clear()

            samples = []
            for _ in range(30):
                iteration()
            for _ in range(9):
                gc.collect()
                start = time.process_time_ns()
                for _ in range(1000):
                    iteration()
                samples.append((time.process_time_ns() - start) / 1000)
            if not hit:
                assert len(store.last[0]) == 64
            outputs[variant] = store.last
            results.append(
                {
                    "variant": variant,
                    "remote_hit": hit,
                    "kv_events": enabled,
                    "requests_per_sample": 1000,
                    "keys_per_request": 64,
                    "cpu_ns_per_request": samples,
                    "median_cpu_ns_per_request": statistics.median(samples),
                }
            )
        assert outputs["baseline"] == outputs["patched"]
print(
    json.dumps(
        {
            "level": "local CPU microbenchmark; store and GPU are stubs",
            "results": results,
        },
        indent=2,
    )
)
