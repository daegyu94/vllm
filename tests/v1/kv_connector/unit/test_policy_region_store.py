# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/policy_region.py"
)
spec = importlib.util.spec_from_file_location("policy_region_standalone", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Fake:
    def __init__(self):
        self.calls = []
        self.keys = {}
        self.fail = False
        self.next = 0

    def join(self, p):
        self.calls.append(("join", p))

    def leave(self, p):
        self.calls.append(("leave", p))

    def reserve(self, p, nonce, size, slots):
        self.calls.append(("reserve", p))
        self.next += 1
        return NS(id=self.next, policy=p, object_size=size, slots=slots)

    def put(self, r, keys, ptrs, sizes):
        if self.fail:
            raise RuntimeError("uncertain publication")
        self.keys.update(dict.fromkeys(keys, True))
        return [0] * len(keys)

    def acquire(self, p, keys):
        self.next += 1
        self.calls.append(("acquire", p))
        objects = [NS(found=k in self.keys) for k in keys]
        return NS(
            read_id=self.next if any(o.found for o in objects) else 0, objects=objects
        )

    def release(self, v):
        self.calls.append(("release", v))

    def close_region(self, r):
        self.calls.append(("close_region", r))

    def flush(self):
        self.calls.append(("flush",))

    def get(self, v, keys, pointers, sizes):
        return [sum(s) for s in sizes]

    def close(self):
        self.calls.append(("close",))


class Tests(unittest.TestCase):
    def make(self, limit=8):
        n = Fake()
        s = module.PolicyRegionStore("job/initial", slots=4, cache_keys=limit, native=n)
        return n, s

    def test_geometry_and_bucketing(self):
        n, s = self.make()
        self.assertEqual(
            s.batch_put_from_multi_buffers(["a", "b"], [[1], [2]], [[4], [4]], None),
            [0, 0],
        )
        s.batch_put_from_multi_buffers(["c"], [[3]], [[8]], None)
        self.assertEqual(sum(c[0] == "reserve" for c in n.calls), 2)

    def test_descriptor_reuse(self):
        n, s = self.make()
        n.keys = {"a": True}
        self.assertEqual(s.batch_is_exist(["a", "miss"]), [1, 0])
        self.assertEqual(s.batch_get_into_multi_buffers(["a"], [[1]], [[4]]), [4])
        self.assertEqual(sum(c[0] == "acquire" for c in n.calls), 1)
        n.keys["miss"] = True
        self.assertEqual(s.batch_is_exist(["miss"]), [1])

    def test_cache_bound(self):
        n, s = self.make(1)
        n.keys = {"a": True, "b": True}
        s.batch_is_exist(["a"])
        s.batch_is_exist(["b"])
        self.assertLessEqual(len(s._keys), 1)
        self.assertEqual(sum(c[0] == "release" for c in n.calls), 1)

    def test_reset_does_not_retire_policy(self):
        n, s = self.make()
        s.remove_all(force=True)
        self.assertNotIn(("leave", "job/initial"), n.calls)
        s.batch_is_exist(["a"])

    def test_transition_and_failure_fence(self):
        n, s = self.make()
        s.transition_policy("job/weights/1")
        self.assertLess(
            n.calls.index(("flush",)), n.calls.index(("leave", "job/initial"))
        )
        self.assertEqual(n.calls[-1], ("join", "job/weights/1"))
        with self.assertRaises(ValueError):
            s.transition_policy("job/initial")
        n.fail = True
        with self.assertRaises(RuntimeError):
            s.batch_put_from_multi_buffers(["a"], [[1]], [[4]], None)
        with self.assertRaises(RuntimeError):
            s.batch_is_exist(["a"])

    def test_invalid_contracts(self):
        with self.assertRaises(ValueError):
            module.PolicyRegionStore("", native=Fake())
        n, s = self.make()
        with self.assertRaises(ValueError):
            s.batch_put_from_multi_buffers(["a"], [], [], None)
        s.close()
        s.close()
        with self.assertRaises(RuntimeError):
            s.batch_is_exist(["a"])


class QueueFenceTests(unittest.TestCase):
    def test_real_worker_hook_drains_send_and_receive(self):
        import ast
        import queue
        import sys
        import threading

        source = SOURCE.with_name("worker.py")
        tree = ast.parse(source.read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MooncakeStoreWorker"
        )
        fn = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "transition_policy_region"
        )
        scope: dict[str, Any] = {"__package__": "poc_store"}
        sys.modules["poc_store.policy_region"] = module
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                str(source),
                "exec",
            ),
            scope,
        )
        native = Fake()
        store = module.PolicyRegionStore("job/initial", native=native)
        send: queue.Queue[int] = queue.Queue()
        receive: queue.Queue[int] = queue.Queue()
        send.put(1)
        receive.put(1)
        obj = NS(
            store=store,
            kv_send_thread=NS(request_queue=send),
            kv_recv_threads=[NS(request_queue=receive)],
        )
        done = threading.Event()
        errors = []

        def call():
            try:
                scope["transition_policy_region"](obj, "job/weights/1")
            except Exception as e:
                errors.append(e)
            finally:
                done.set()

        worker = threading.Thread(target=call)
        worker.start()
        self.assertFalse(done.wait(0.05))
        send.task_done()
        self.assertFalse(done.wait(0.05))
        receive.task_done()
        self.assertTrue(done.wait(2))
        worker.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.policy, "job/weights/1")


if __name__ == "__main__":
    unittest.main()
