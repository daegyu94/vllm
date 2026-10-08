# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests without gated model/tokenizer initialization."""

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore


def test_prepare_cache_reset_waits_for_every_engine():
    from vllm.v1.engine.core_client import AsyncMPClient

    async def run():
        client = AsyncMPClient.__new__(AsyncMPClient)
        client.core_engines = [b"dp0", b"dp1"]
        entered = set()
        release = asyncio.Event()

        async def call(method, *args, engine):
            assert (method, args) == (
                "collective_rpc",
                ("prepare_kv_cache_reset", None, (), {}),
            )
            entered.add(engine)
            await release.wait()

        client._call_utility_async = call
        prepared = asyncio.create_task(client.prepare_kv_cache_reset_async())
        for _ in range(10):
            if len(entered) == 2:
                break
            await asyncio.sleep(0)
        assert entered == {b"dp0", b"dp1"}
        assert not prepared.done()
        release.set()
        await prepared

    asyncio.run(run())


@pytest.mark.parametrize("admitted", [False, True])
def test_core_prepares_every_worker_before_external_reset(admitted):
    core = EngineCore.__new__(EngineCore)
    core.scheduler = MagicMock()
    core.model_executor = MagicMock()
    core.scheduler.has_unfinished_requests.return_value = not admitted
    events = []

    def reset(running, connector):
        events.append(("reset", running, connector))
        return admitted

    core.scheduler.reset_prefix_cache.side_effect = reset
    core.model_executor.collective_rpc.side_effect = lambda method: events.append(
        ("workers", method)
    )
    assert core.reset_prefix_cache(reset_connector=True) is admitted
    if admitted:
        assert events == [("workers", "prepare_kv_cache_reset"), ("reset", False, True)]
    else:
        assert events == []
