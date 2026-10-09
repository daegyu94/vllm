# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol tests without gated model/tokenizer initialization."""

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.v1.core.sched.scheduler import PauseState
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
    core.batch_queue = None
    core.model_executor = MagicMock()
    core.scheduler.can_reset_prefix_cache.return_value = admitted
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


@pytest.mark.parametrize("reset_store", [False, True])
@pytest.mark.parametrize("local_ok", [False, True])
def test_shared_reset_clears_all_dp_engines_before_one_deletion(reset_store, local_ok):
    from vllm.v1.engine.core_client import AsyncMPClient

    async def run():
        client = AsyncMPClient.__new__(AsyncMPClient)
        client.core_engines = [b"dp0", b"dp1"]
        cleared, deleted = set(), []
        release, entered = asyncio.Event(), asyncio.Event()

        async def call(method, *args, engine):
            if method == "reset_local_prefix_cache_for_shared_store":
                if engine == b"dp1":
                    entered.set()
                    await release.wait()
                cleared.add(engine)
                return local_ok
            assert cleared == set(client.core_engines)
            assert (method, args) == ("reset_prefix_cache", (False, True))
            deleted.append(engine)
            return True

        client._call_utility_async = call
        task = asyncio.create_task(client.reset_shared_prefix_cache_async(reset_store))
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not deleted
        release.set()
        assert await task is local_ok
        assert deleted == ([b"dp0"] if reset_store and local_ok else [])

    asyncio.run(run())


@pytest.mark.parametrize("safe", [False, True])
def test_local_shared_reset_retains_core_lifetime_checks(safe):
    core = EngineCore.__new__(EngineCore)
    core.batch_queue = None
    core.scheduler = MagicMock()
    core.model_executor = MagicMock()
    core.scheduler.pause_state = PauseState.PAUSED_ALL
    core.scheduler.can_reset_prefix_cache.return_value = safe
    core.scheduler.reset_prefix_cache.return_value = safe
    assert core.reset_local_prefix_cache_for_shared_store() is safe
    if safe:
        core.model_executor.collective_rpc.assert_called_once_with(
            "prepare_kv_cache_reset"
        )
        core.scheduler.reset_prefix_cache.assert_called_once_with(False, False)
        core.scheduler.get_kv_connector().reset_cache_local.assert_called_once()
    else:
        core.model_executor.collective_rpc.assert_not_called()
        core.scheduler.reset_prefix_cache.assert_not_called()


@pytest.mark.parametrize(
    "pause_state", [PauseState.UNPAUSED, PauseState.PAUSED_NEW, PauseState.PAUSED_ALL]
)
def test_shared_sleep_keeps_local_lifetime_checks_and_avoids_deletion(pause_state):
    core = EngineCore.__new__(EngineCore)
    core.scheduler = MagicMock()
    core.batch_queue = None
    core.model_executor = MagicMock()
    core.scheduler.pause_state = pause_state
    core.mm_receiver_cache = None
    core.scheduler.can_reset_prefix_cache.return_value = True
    core.scheduler.reset_prefix_cache.return_value = True

    def set_pause(state):
        core.scheduler.pause_state = state

    core.scheduler.set_pause_state.side_effect = set_pause

    def prepare(method):
        assert core.scheduler.pause_state == PauseState.PAUSED_ALL

    core.model_executor.collective_rpc.side_effect = prepare
    if pause_state != PauseState.UNPAUSED:
        core.sleep_after_shared_cache_reset(1)
        assert core.scheduler.pause_state == PauseState.PAUSED_ALL
        core.scheduler.reset_prefix_cache.assert_called_once_with(False, False)
        core.scheduler.get_kv_connector().reset_cache.assert_not_called()
        core.model_executor.sleep.assert_called_once_with(1)
    else:
        with pytest.raises(RuntimeError, match="paused, idle"):
            core.sleep_after_shared_cache_reset(1)
        core.scheduler.reset_prefix_cache.assert_not_called()
        core.model_executor.sleep.assert_not_called()
