# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in DFS-only policy regions. Generation must be paused during transition.

Trusted clients retain physical ranges through writer/read leases. No timeout
recycles an uncertain range. This prototype deliberately has no HA recovery.
"""

import threading
import uuid
from collections import OrderedDict


class PolicyRegionStore:
    def __init__(self, policy, slots=64, cache_keys=4096, native=None):
        if not policy or slots < 1 or cache_keys < 1:
            raise ValueError("policy, slots and cache_keys must be nonempty/positive")
        self.policy = policy
        self.slots = slots
        self.cache_keys = cache_keys
        self.native = native
        self._ready = native is not None
        self._closed = False
        self._retired = set()
        self._regions = {}
        self._views = OrderedDict()
        self._keys = {}
        self._lock = threading.RLock()
        if native is not None:
            native.join(policy)

    def setup(
        self,
        hostname,
        metadata,
        global_size,
        local_size,
        protocol,
        device,
        master,
        **kwargs,
    ):
        if protocol != "tcp" or kwargs.get("tenant_id"):
            raise ValueError("policy region PoC requires TCP and the default tenant")
        from policy_region_native import Client

        self.native = Client(master, hostname, True, 0)
        self.native.join(self.policy)
        self._ready = True
        return 0

    def _ensure_ready(self):
        if self._closed or not self._ready:
            raise RuntimeError("policy region store is closed or fenced")

    def register_buffer(self, pointer, size):
        with self._lock:
            self._ensure_ready()
            self.native.register_buffer(pointer, size)
            return 0

    def _retain(self, keys, view):
        positive = [k for k, loc in zip(keys, view.objects) if loc.found]
        if not view.read_id:
            return False
        if len(positive) > self.cache_keys:
            return False
        while self._views and len(self._keys) + len(positive) > self.cache_keys:
            read_id, (_, old_keys) = next(iter(self._views.items()))
            self.native.release(read_id)
            self._views.pop(read_id)
            for key in old_keys:
                if self._keys.get(key) == read_id:
                    self._keys.pop(key)
        self._views[view.read_id] = (view, positive)
        self._keys.update(dict.fromkeys(positive, view.read_id))
        return True

    def batch_is_exist(self, keys):
        with self._lock:
            self._ensure_ready()
            missing = list(dict.fromkeys(k for k in keys if k not in self._keys))
            result = dict.fromkeys(keys, 1)
            if missing:
                view = self.native.acquire(self.policy, missing)
                result.update({k: int(o.found) for k, o in zip(missing, view.objects)})
                if not self._retain(missing, view) and view.read_id:
                    self.native.release(view.read_id)
            return [result[k] for k in keys]

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, replica=None):
        with self._lock:
            self._ensure_ready()
            if len(keys) != len(pointers) or len(keys) != len(sizes):
                raise ValueError("batch length mismatch")
            grouped: dict[int, list[int]] = {}
            for i, (p, s) in enumerate(zip(pointers, sizes)):
                if not s or len(p) != len(s) or any(n <= 0 for n in s):
                    raise ValueError("invalid object slices")
                grouped.setdefault(sum(s), []).append(i)
            result = [0] * len(keys)
            try:
                for size, indices in grouped.items():
                    while indices:
                        region, used = self._regions.get(size, (None, 0))
                        if region is None or used == self.slots:
                            if region is not None:
                                self.native.close_region(region.id)
                            region = self.native.reserve(
                                self.policy, uuid.uuid4().hex, size, self.slots
                            )
                            used = 0
                        selected, indices = (
                            indices[: self.slots - used],
                            indices[self.slots - used :],
                        )
                        values = self.native.put(
                            region,
                            [keys[i] for i in selected],
                            [pointers[i] for i in selected],
                            [sizes[i] for i in selected],
                        )
                        for i, value in zip(selected, values):
                            result[i] = value
                        self._regions[size] = (region, used + len(selected))
            except Exception:
                self._ready = False
                raise
            return result

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        with self._lock:
            self._ensure_ready()
            if len(keys) != len(pointers) or len(keys) != len(sizes):
                raise ValueError("batch length mismatch")
            result = [0] * len(keys)
            groups: dict[int, list[int]] = {}
            missing = []
            for i, key in enumerate(keys):
                read_id = self._keys.get(key)
                if read_id is None:
                    missing.append(i)
                else:
                    groups.setdefault(read_id, []).append(i)
                    self._views.move_to_end(read_id)
            for read_id, indices in groups.items():
                values = self.native.get(
                    self._views[read_id][0],
                    [keys[i] for i in indices],
                    [pointers[i] for i in indices],
                    [sizes[i] for i in indices],
                )
                for i, value in zip(indices, values):
                    result[i] = value
            if missing:
                names = [keys[i] for i in missing]
                view = self.native.acquire(self.policy, names)
                try:
                    values = self.native.get(
                        view,
                        names,
                        [pointers[i] for i in missing],
                        [sizes[i] for i in missing],
                    )
                    for i, value in zip(missing, values):
                        result[i] = value
                finally:
                    if view.read_id:
                        self.native.release(view.read_id)
            return result

    def remove_all(self, force=False):
        with self._lock:
            self._ensure_ready()
            # Prefix reset/wake-up may occur without a weight change. Keep
            # membership and identity; only explicit transition retires policy.
            self.native.flush()
            self._regions.clear()
            self._views.clear()
            self._keys.clear()
            return 0

    def transition_policy(self, policy):
        with self._lock:
            if self._closed or not policy or policy in self._retired:
                raise ValueError("invalid or retired policy identity")
            if policy == self.policy and self._ready:
                return True
            self._ready = False
            self.native.flush()
            self.native.leave(self.policy)
            self._retired.add(self.policy)
            self.native.join(policy)
            self._regions.clear()
            self._views.clear()
            self._keys.clear()
            self.policy = policy
            self._ready = True
            return True

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._ready = False
            if self.native is not None:
                self.native.close()
            self._closed = True
