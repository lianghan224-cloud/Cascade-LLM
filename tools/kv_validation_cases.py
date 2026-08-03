"""Executable validation scenarios for the KV remediation matrix."""

from concurrent.futures import CancelledError
import random
import threading
import time

import torch

from layer_streaming.attention.paged import (
    PagedWorkload,
    ReferencePagedExactBackend,
    classify_paged_workload,
    default_paged_registry,
)
from layer_streaming.kv import (
    KVCapacityError,
    KVLifecycleError,
    KVPagePoolV1,
    PagedKVRuntime,
    TorchPagedKVKernelBackend,
)
from layer_streaming.kv.scheduler import TieredAttentionCoordinator
from layer_streaming.kv.selection import LogicalKVBlockId, QuestCPUIndex
from layer_streaming.kv.stores import (
    KVTier,
    PrefetchCancelled,
    TieredKVStore,
)
from layer_streaming.kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
)
from layer_streaming.providers.base import PagedProviderBundle


LIFECYCLE_IDS = tuple("V{:02d}".format(value) for value in range(2, 16))
SHARING_IDS = tuple("V{:02d}".format(value) for value in range(16, 27))
INDEX_IDS = tuple("V{:02d}".format(value) for value in range(27, 39))
TIER_IDS = tuple("V{:02d}".format(value) for value in range(39, 52))
SCHEDULER_IDS = tuple("V{:02d}".format(value) for value in range(52, 56))
ROUTING_IDS = tuple("V{:02d}".format(value) for value in range(56, 60))


def _metrics(ids, **values):
    return {case_id: dict(values) for case_id in ids}


def minimize_sequence(sequence, fails):
    """Simple deterministic one-element delta debugging."""

    sequence = list(sequence)
    changed = True
    while changed and len(sequence) > 1:
        changed = False
        for index in range(len(sequence)):
            candidate = sequence[:index] + sequence[index + 1 :]
            if candidate and fails(candidate):
                sequence = candidate
                changed = True
                break
    return sequence


def run_lifecycle_suite(seed=20260803, operations=100000):
    # Explicit underflow, double release, ABA and bounded quiescence checks.
    probe = KVPagePoolV1(2, "mock", "bf16")
    handle = probe.allocate(owner_hint=1)
    probe.seal(handle, 1)
    probe.pin(handle)
    probe.unpin(handle)
    try:
        probe.unpin(handle)
        raise AssertionError("V05 double unpin was accepted")
    except KVLifecycleError:
        pass
    probe.release(handle)
    try:
        probe.release(handle)
        raise AssertionError("V05 double release was accepted")
    except KVLifecycleError:
        pass
    stale = handle
    replacement = probe.allocate(owner_hint=2)
    assert stale.page_id == replacement.page_id
    assert stale.generation != replacement.generation
    try:
        probe.descriptor(stale)
        raise AssertionError("V06 stale generation was accepted")
    except KVLifecycleError:
        pass
    probe.release(replacement)

    wait_pool = KVPagePoolV1(1, "wait", "bf16")
    waiting = wait_pool.allocate()
    wait_pool.seal(waiting, 1)
    wait_pool.pin(waiting)
    releaser = threading.Thread(
        target=lambda: (time.sleep(0.01), wait_pool.unpin(waiting))
    )
    releaser.start()
    wait_pool.wait_quiescent(timeout_seconds=1.0)
    releaser.join()
    wait_pool.release(waiting)

    # OOM leaves the first allocation intact and subsequent release usable.
    capacity = KVPagePoolV1(1, "capacity", "bf16")
    only = capacity.allocate()
    try:
        capacity.allocate()
        raise AssertionError("V09 capacity failure was not raised")
    except KVCapacityError:
        pass
    capacity.release(only)
    assert capacity.free_pages == 1

    # Compute failure returns its pin through the context-manager path.
    kernel = KVPagePoolV1(1, "kernel", "bf16")
    compute = kernel.allocate()
    kernel.seal(compute, 1)
    try:
        with kernel.pinned((compute,)):
            raise RuntimeError("injected kernel submit failure")
    except RuntimeError:
        pass
    assert kernel.descriptor(compute).pin_count == 0
    kernel.release(compute)

    # IO submit/completion failures retain the source authority.
    with TieredKVStore() as io_store:
        io_store.put("io", b"source", KVTier.GPU, version=1)
        for stage in ("submit", "completion"):
            try:
                io_store.migrate("io", KVTier.CPU, failure_stage=stage)
                raise AssertionError("V11/V12 injected IO failure did not fire")
            except IOError:
                pass
            assert io_store.get("io") == b"source"
            assert io_store.record("io").authoritative_tier == KVTier.GPU

    # Multiple independent workers stress allocator lock and generation reuse.
    concurrent = KVPagePoolV1(16, "threads", "bf16")
    shared_anchor = concurrent.allocate(owner_hint=-1)
    concurrent.seal(shared_anchor, 1)
    thread_errors = []

    def worker(worker_id):
        try:
            for _ in range(500):
                concurrent.retain(shared_anchor)
                while True:
                    try:
                        item = concurrent.allocate(owner_hint=worker_id)
                        break
                    except KVCapacityError:
                        time.sleep(0)
                concurrent.seal(item, 1)
                concurrent.pin(item)
                concurrent.unpin(item)
                concurrent.release(item)
                concurrent.release(shared_anchor)
        except BaseException as exc:
            thread_errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not thread_errors, thread_errors
    concurrent.release(shared_anchor)
    concurrent.validate_invariants()
    assert concurrent.free_pages == concurrent.page_count

    # Reproducible 100k primitive lifecycle operations. Each randomized mini
    # transaction is bounded, preventing the test harness itself from growing
    # an unbounded reference list while still exercising generation reuse.
    rng = random.Random(int(seed))
    pool = KVPagePoolV1(8, "random", "bf16")
    executed = 0
    while executed < int(operations):
        item = pool.allocate(owner_hint=rng.randrange(32))
        executed += 1
        pool.seal(item, rng.randrange(1, 17))
        executed += 1
        choice = rng.random()
        if choice < 0.34:
            pool.pin(item)
            pool.unpin(item)
            executed += 2
        elif choice < 0.67:
            pool.retain(item)
            pool.release(item)
            executed += 2
        else:
            target = pool.allocate(owner_hint=99)
            pool.begin_copy(item, target)
            pool.end_copy(item, target, 1)
            pool.release(target)
            executed += 4
        pool.release(item)
        executed += 1
        if executed % 1000 < 7:
            pool.validate_invariants()
    pool.validate_invariants()
    assert pool.free_pages == pool.page_count

    def underflows(sequence):
        local = KVPagePoolV1(1, "repro", "bf16")
        item = local.allocate()
        local.seal(item, 1)
        try:
            for operation in sequence:
                if operation == "pin":
                    local.pin(item)
                elif operation == "unpin":
                    local.unpin(item)
            return False
        except KVLifecycleError:
            return True

    minimized = minimize_sequence(
        ["pin", "unpin", "pin", "unpin", "unpin"], underflows
    )
    assert minimized == ["unpin"]
    return _metrics(
        LIFECYCLE_IDS,
        seed=int(seed),
        random_operations=executed,
        free_pages=pool.free_pages,
        concurrent_cycles=2000,
        minimized_repro=minimized,
    )


def _policy(reuse=KVReusePolicy.REQUEST_ONLY, selection=KVSelectionPolicy.DENSE):
    return KVPolicy(
        accuracy=(KVAccuracy.SPARSE if selection != KVSelectionPolicy.DENSE else KVAccuracy.EXACT),
        dtype=KVDataType.BF16,
        selection=selection,
        reuse=reuse,
        attention_backend="reference_paged_exact",
        page_size=16,
        page_budget=(2 if selection != KVSelectionPolicy.DENSE else 0),
    )


def _runtime(reuse=KVReusePolicy.REQUEST_ONLY, selection=KVSelectionPolicy.DENSE):
    return PagedKVRuntime(
        layer_count=2,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        page_count=32,
        page_size=16,
        dtype=torch.bfloat16,
        device="cpu",
        policy=_policy(reuse=reuse, selection=selection),
        allow_reference=True,
    )


def _append(runtime, state, count, offset=0):
    values = torch.arange(
        offset * 4,
        (offset + count) * 4,
        dtype=torch.float32,
    ).reshape(count, 1, 4).to(runtime.dtype)
    for layer in range(runtime.layer_count):
        runtime.append((state,), layer, values + layer * 1000, values + 100, (count,))
    return values


def run_sharing_suite(seed=20260803):
    torch.manual_seed(int(seed) % (2 ** 31))
    runtime = _runtime()
    parent = runtime.create_request(128, request_id=1)
    original = _append(runtime, parent, 18)
    child = runtime.fork(parent, request_id=2)
    shared = parent.block_table.handles[0]
    assert runtime.page_pool.descriptor(shared).ref_count == 2
    _append(runtime, child, 3, offset=6)
    assert parent.block_table.handles[1] != child.block_table.handles[1]
    parent_page, _ = runtime.store.read_pages(0, (parent.block_table.handles[1].page_id,))
    assert torch.equal(parent_page[0, 0, :2], original[16:18, 0])
    runtime.release(parent)
    query = torch.ones(1, 2, 4, dtype=runtime.dtype)
    runtime.attend((child,), 0, query, (1,), phase="decode")

    beam_a = runtime.fork(child, request_id=3)
    beam_b = runtime.fork(child, request_id=4)
    runtime.release(beam_a)
    assert beam_b.sequence_length == child.sequence_length
    runtime.release(beam_b)

    draft_start = child.sequence_length
    _append(runtime, child, 6, offset=draft_start)
    runtime.commit_speculative(child, draft_start, 2)
    assert child.sequence_length == draft_start + 2
    assert child.tail_valid_tokens == (child.sequence_length % 16 or 16)
    runtime.rollback(child, 3)
    assert child.sequence_length == 3
    assert len(child.block_table.handles) == 1
    assert child.layer_lengths == [3, 3]
    runtime.close()
    assert runtime.page_pool.free_pages == runtime.page_pool.page_count

    prefix_runtime = _runtime(reuse=KVReusePolicy.PREFIX_MEMORY)
    origin = prefix_runtime.create_request(64, request_id=10, reuse_namespace="tenant")
    _append(prefix_runtime, origin, 32)
    prefix_runtime.register_prefix(origin, list(range(32)))
    reused, match = prefix_runtime.reuse_prefix(
        list(range(32)), 64, reuse_namespace="tenant", request_id=11
    )
    assert match.matched_tokens == 32
    prefix_runtime.release(origin)
    evicted = prefix_runtime.prefix_cache.evict_all()
    assert evicted == 2
    assert reused.sequence_length == 32
    prefix_runtime.release(reused)
    prefix_runtime.close()
    assert prefix_runtime.page_pool.free_pages == prefix_runtime.page_pool.page_count
    return _metrics(
        SHARING_IDS,
        seed=int(seed),
        cow_isolated=True,
        partial_commit_length=draft_start + 2,
        rollback_length=3,
        prefix_evicted_pages=evicted,
    )


def _quest_fixture():
    index = QuestCPUIndex()
    records = []
    for logical in range(5):
        records.append(
            index.build(
                [
                    [logical + 0.25, logical + 1.0],
                    [logical + 0.50, logical + 2.0],
                ],
                {
                    "logical_block_id": LogicalKVBlockId(
                        "model", "session", "branch", 0, logical
                    ),
                    "token_start": logical * 2,
                    "data_version": 7,
                },
            )
        )
    return index, records


def run_index_suite(seed=20260803):
    del seed
    index, records = _quest_fixture()
    full = index.select([1.0, -0.5], records[::-1], mode="full")
    assert [item.logical_block_id.logical_block for item in full.records] == list(range(5))
    reconstructed = tuple(row for record in full.records for row in record.rows)
    reference = tuple(row for record in records for row in record.rows)
    assert reconstructed == reference
    exact_scores = {
        record.record_id.value: max(
            sum(q * k for q, k in zip([1.0, -0.5], row))
            for row in record.rows
        )
        for record in records
    }
    sparse = index.select(
        [1.0, -0.5], records, budget=2, mode="budget", exact_scores=exact_scores
    )
    assert sparse.selected_count == 2 and sparse.recall is not None
    appended = index.update_append(records[0], [[9.0, 10.0]], new_version=8)
    assert appended.token_count == 3 and appended.index_version == 8
    shared = index.fork_ref(records[1])
    assert shared is records[1]
    clone = index.cow_clone(records[1])
    changed = index.update_append(clone, [[30.0, 40.0]], new_version=8)
    assert changed.token_count == 3 and records[1].token_count == 2
    rolled = index.rollback(changed, 1, target_version=9)
    assert rolled.token_count == 1 and rolled.index_version == 9
    stale = index.invalidate_for_test(records[2], 6)
    try:
        index.select([1.0, -0.5], (stale,), budget=1, mode="budget")
        raise AssertionError("V36 stale index was selected")
    except KVLifecycleError:
        pass
    restored = index.deserialize(index.serialize(records[3]))
    assert restored == records[3]
    assert index.validate(
        restored,
        {
            "logical_block_id": restored.logical_block_id,
            "data_version": 7,
            "token_count": 2,
        },
    )
    empty = index.build(
        [],
        {
            "logical_block_id": LogicalKVBlockId("m", "s", "b", 0, 99),
            "token_start": 0,
            "data_version": 1,
        },
    )
    assert index.select([], (empty,), mode="full").selected_count == 1
    stats = index.stats()
    assert stats["queries"] >= 3 and stats["candidates"] >= 11
    return _metrics(
        INDEX_IDS,
        candidate_count=sparse.candidate_count,
        selected_count=sparse.selected_count,
        recall=sparse.recall,
        max_score_error=sparse.max_score_error,
        serialized_bytes=len(index.serialize(records[0])),
    )


def _wait_clean(store, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = store.stats()
        if not stats["inflight_io"] and not stats["reservations"]:
            return stats
        time.sleep(0.005)
    raise TimeoutError("tiered store did not quiesce")


def run_tier_suite(seed=20260803):
    del seed
    payload = bytes(range(128))
    with TieredKVStore(io_delay=0.02) as store:
        store.put("block", payload, KVTier.GPU, version=3)
        assert store.record("block").authoritative_tier == KVTier.GPU
        store.migrate("block", KVTier.CPU)
        store.migrate("block", KVTier.SSD)
        assert store.get("block") == payload
        assert store.record("block").authoritative_tier == KVTier.SSD

        store.evict("block", KVTier.GPU)
        migration_error = []

        def migrate_gpu():
            try:
                store.migrate("block", KVTier.GPU)
            except BaseException as exc:
                migration_error.append(exc)

        thread = threading.Thread(target=migrate_gpu)
        thread.start()
        time.sleep(0.005)
        assert store.get("block") == payload
        assert store.record("block").authoritative_tier == KVTier.SSD
        thread.join()
        assert not migration_error
        assert store.get("block", KVTier.GPU) == payload

        for stage, target in (("submit", KVTier.CPU), ("completion", KVTier.GPU)):
            if store.is_resident("block", target):
                store.evict("block", target)
            authority = store.record("block").authoritative_tier
            try:
                store.migrate("block", target, failure_stage=stage)
                raise AssertionError("V45 migration failure was not injected")
            except IOError:
                pass
            assert store.record("block").authoritative_tier == authority
            assert store.get("block") == payload

        if not store.is_resident("block", KVTier.GPU):
            store.migrate("block", KVTier.GPU)
        store.pin("block")
        try:
            store.evict("block", KVTier.GPU)
            raise AssertionError("V46 pinned eviction succeeded")
        except KVLifecycleError:
            pass
        store.unpin("block")
        store.evict("block", KVTier.GPU)
        first = store.prefetch("block", KVTier.GPU)
        second = store.prefetch("block", KVTier.GPU)
        assert first is second
        first.result(timeout=2.0)
        assert store.stats()["prefetch_deduplicated"] >= 1

        store.evict("block", KVTier.GPU)
        cancelled = store.prefetch("block", KVTier.GPU)
        assert store.cancel_prefetch("block", KVTier.GPU)
        try:
            cancelled.result(timeout=2.0)
        except (CancelledError, PrefetchCancelled):
            pass
        clean = _wait_clean(store)
        assert clean["pins"] == 0 and clean["pending_prefetches"] == 0

        if not store.is_resident("block", KVTier.GPU):
            store.prefetch("block", KVTier.GPU).result(timeout=2.0)
        store.evict("block", KVTier.GPU)
        store.prefetch("block", KVTier.GPU).result(timeout=2.0)
        assert store.get("block", KVTier.GPU) == payload
        store.validate_record("block")
        metrics = store.stats()

    with TieredKVStore(capacities={KVTier.GPU: 1}) as limited:
        limited.put("one", b"one", KVTier.GPU)
        try:
            limited.put("two", b"two", KVTier.GPU)
            raise AssertionError("V49 capacity limit was ignored")
        except KVCapacityError:
            pass
    return _metrics(
        TIER_IDS,
        roundtrip_bytes=len(payload),
        migrations=metrics["migrations"],
        failures=metrics["failures"],
        deduplicated=metrics["prefetch_deduplicated"],
        cancellations=metrics["prefetch_cancelled"],
        authoritative_unique=True,
    )


def run_scheduler_suite(seed=20260803):
    del seed
    with TieredKVStore(io_delay=0.01) as store:
        quest = QuestCPUIndex()
        indexed_ids = []
        records = []
        for index in range(2):
            logical = LogicalKVBlockId("m", "s", "q", 0, index)
            indexed_ids.append(logical)
            records.append(
                quest.build(
                    [[float(index), 1.0]],
                    {
                        "logical_block_id": logical,
                        "token_start": index,
                        "data_version": 1,
                    },
                )
            )
            store.put(logical, bytes([index]) * 16, KVTier.CPU)
        for index in range(4):
            store.put("s{}".format(index), bytes([index]) * 16, KVTier.CPU)
        coordinator = TieredAttentionCoordinator(store)
        selection = quest.select([1.0, 1.0], records, mode="full")
        result = coordinator.execute(
            selection, lambda view: b"".join(view.payloads)
        )
        assert len(result) == 32
        assert store.stats()["pins"] == 0
        try:
            coordinator.execute(
                ("s0",),
                lambda view: (_ for _ in ()).throw(RuntimeError("compute")),
            )
        except RuntimeError:
            pass
        assert store.stats()["pins"] == 0

        cancel_event = threading.Event()
        cancel_event.set()
        try:
            coordinator.prepare(("s2",), cancel_event=cancel_event)
            raise AssertionError("V55 cancelled request continued")
        except PrefetchCancelled:
            pass

        order = []
        errors = []

        def scheduled(name):
            try:
                coordinator.execute(
                    (name,), lambda view: order.append(name)
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=scheduled, args=("s{}".format(i),)) for i in range(4)]
        for thread in threads:
            thread.start()
            time.sleep(0.002)
        for thread in threads:
            thread.join()
        assert not errors and set(order) == {"s0", "s1", "s2", "s3"}
        assert coordinator.stats()["served"] == coordinator.stats()["requests"]
        metrics = coordinator.stats()
        assert store.stats()["pins"] == 0
    return _metrics(SCHEDULER_IDS, **metrics)


class _DecodeSuffixReference(ReferencePagedExactBackend):
    name = "decode_suffix_reference"
    workload_kinds = ("decode", "short_suffix")


def run_routing_suite(seed=20260803):
    torch.manual_seed(int(seed) % (2 ** 31))
    registry = default_paged_registry(load_cuda=False)
    registry.register(
        PagedProviderBundle(
            name="decode_suffix_reference",
            attention_backend=_DecodeSuffixReference(),
            kv_kernel_backend=TorchPagedKVKernelBackend(),
        )
    )
    policy = KVPolicy(
        dtype=KVDataType.BF16,
        page_size=16,
        attention_backend="decode_suffix_reference",
    )
    runtime = PagedKVRuntime(
        layer_count=1,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        page_count=8,
        page_size=16,
        dtype=torch.bfloat16,
        device="cpu",
        policy=policy,
        provider_registry=registry,
        allow_reference=True,
    )
    state = runtime.create_request(32)
    values = torch.randn(20, 1, 4, dtype=torch.bfloat16)
    runtime.append((state,), 0, values, values, (20,))

    full_query = torch.randn(20, 2, 4, dtype=torch.bfloat16)
    runtime.attend((state,), 0, full_query, (20,), phase="full_prefill")
    full_decision = dict(runtime.dispatcher.last_decision)
    assert full_decision["selected"] == "reference_paged_exact"
    assert full_decision["workload"] == PagedWorkload.FULL_PREFILL.value

    decode_query = torch.randn(1, 2, 4, dtype=torch.bfloat16)
    runtime.attend((state,), 0, decode_query, (1,), phase="decode")
    decode_decision = dict(runtime.dispatcher.last_decision)
    assert decode_decision["selected"] == "decode_suffix_reference"

    chunk_query = torch.randn(5, 2, 4, dtype=torch.bfloat16)
    runtime.attend((state,), 0, chunk_query, (5,), phase="chunked_prefill")
    chunk_decision = dict(runtime.dispatcher.last_decision)
    assert chunk_decision["selected"] == "reference_paged_exact"

    suffix_query = torch.randn(2, 2, 4, dtype=torch.bfloat16)
    runtime.attend((state,), 0, suffix_query, (2,), phase="short_suffix")
    suffix_decision = dict(runtime.dispatcher.last_decision)
    assert suffix_decision["selected"] == "decode_suffix_reference"
    assert "correctness" in full_decision["fallback_reason"]
    runtime.close()
    return {
        "V56": full_decision,
        "V57": decode_decision,
        "V58": chunk_decision,
        "V59": {
            "full_fallback": full_decision["fallback_reason"],
            "chunked_fallback": chunk_decision["fallback_reason"],
            "short_suffix_selected": suffix_decision["selected"],
        },
    }


LOGIC_SUITES = (
    (LIFECYCLE_IDS, run_lifecycle_suite),
    (SHARING_IDS, run_sharing_suite),
    (INDEX_IDS, run_index_suite),
    (TIER_IDS, run_tier_suite),
    (SCHEDULER_IDS, run_scheduler_suite),
    (ROUTING_IDS, run_routing_suite),
)


def run_cuda_suite(seed=20260803):
    """Synthetic numerical, stream, failure and allocator-drift validation."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    import torch.nn.functional as functional
    from layer_streaming.providers.sm86.paged_attention import (
        SM86PagedAttentionBackend,
        SM86PagedKVKernelBackend,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(int(seed) % (2 ** 31))
    policy = KVPolicy(
        dtype=KVDataType.BF16,
        page_size=16,
        attention_backend=(
            "sm86"
            if torch.cuda.get_device_capability(device) == (8, 6)
            else "generic_cuda"
        ),
    )
    runtime = PagedKVRuntime(
        layer_count=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=128,
        page_count=16,
        page_size=16,
        dtype=torch.bfloat16,
        device=device,
        policy=policy,
    )
    state = runtime.create_request(256)
    key = torch.randn(17, 2, 128, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    runtime.append((state,), 0, key, value, (17,))

    query_decode = torch.randn(1, 4, 128, dtype=torch.bfloat16, device=device)
    started = time.perf_counter()
    decode = runtime.attend(
        (state,), 0, query_decode, (1,), phase="decode"
    ).output
    torch.cuda.synchronize(device)
    decode_ms = (time.perf_counter() - started) * 1000.0
    expanded_key = key.repeat_interleave(2, dim=1)
    expanded_value = value.repeat_interleave(2, dim=1)
    decode_scores = torch.einsum(
        "thd,shd->ths", query_decode.float(), expanded_key.float()
    ) / (128.0 ** 0.5)
    decode_reference = torch.einsum(
        "ths,shd->thd",
        torch.softmax(decode_scores, dim=-1),
        expanded_value.float(),
    )
    decode_error = float((decode.float() - decode_reference).abs().max().item())
    assert decode_error <= 1.7e-2, decode_error

    query_full = torch.randn(17, 4, 128, dtype=torch.bfloat16, device=device)
    full = runtime.attend(
        (state,), 0, query_full, (17,), phase="full_prefill"
    ).output
    full_reference = functional.scaled_dot_product_attention(
        query_full.transpose(0, 1).unsqueeze(0).float(),
        expanded_key.transpose(0, 1).unsqueeze(0).float(),
        expanded_value.transpose(0, 1).unsqueeze(0).float(),
        dropout_p=0.0,
        is_causal=True,
    ).squeeze(0).transpose(0, 1)
    full_error = float((full.float() - full_reference).abs().max().item())
    assert full_error <= 1.7e-2, full_error
    full_decision = dict(runtime.dispatcher.last_decision)
    assert full_decision["selected"] == "reference_paged_exact"

    query_chunk = torch.randn(5, 4, 128, dtype=torch.bfloat16, device=device)
    positions = torch.arange(12, 17, dtype=torch.int32, device=device)
    chunk = runtime.attend(
        (state,),
        0,
        query_chunk,
        (5,),
        query_positions=(positions,),
        phase="chunked_prefill",
    ).output
    chunk_scores = torch.einsum(
        "thd,shd->ths", query_chunk.float(), expanded_key.float()
    ) / (128.0 ** 0.5)
    mask = torch.arange(17, device=device).view(1, -1) <= positions.view(-1, 1)
    chunk_scores.masked_fill_(~mask.unsqueeze(1), -float("inf"))
    chunk_reference = torch.einsum(
        "ths,shd->thd",
        torch.softmax(chunk_scores, dim=-1),
        expanded_value.float(),
    )
    chunk_error = float((chunk.float() - chunk_reference).abs().max().item())
    assert chunk_error <= 1.7e-2, chunk_error

    streams = [torch.cuda.Stream(device=device) for _ in range(2)]
    stream_outputs = []
    for index in range(8):
        with torch.cuda.stream(streams[index % 2]):
            stream_outputs.append(
                runtime.attend(
                    (state,), 0, query_decode, (1,), phase="decode"
                ).output
            )
    runtime.quiesce(timeout_seconds=10.0)
    for output in stream_outputs:
        assert float((output.float() - decode.float()).abs().max().item()) == 0.0

    # Warm allocator and retain one output on both sides of the measurement.
    measured = runtime.attend(
        (state,), 0, query_decode, (1,), phase="decode"
    ).output
    runtime.quiesce(timeout_seconds=10.0)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    drift_started = time.perf_counter()
    for _ in range(50):
        measured = runtime.attend(
            (state,), 0, query_decode, (1,), phase="decode"
        ).output
    runtime.quiesce(timeout_seconds=10.0)
    torch.cuda.synchronize(device)
    drift_ms = (time.perf_counter() - drift_started) * 1000.0
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    allocated_drift = allocated_after - allocated_before
    reserved_drift = reserved_after - reserved_before
    assert allocated_drift <= 2 * 1024 * 1024, allocated_drift
    assert reserved_drift <= 8 * 1024 * 1024, reserved_drift

    original_bundle = runtime.registry.get(policy.attention_backend)

    class FailingAttention(SM86PagedAttentionBackend):
        def decode(self, request):
            del request
            raise RuntimeError("injected CUDA kernel failure")

    if policy.attention_backend == "sm86":
        runtime.registry.register(
            PagedProviderBundle(
                name="sm86",
                attention_backend=FailingAttention(),
                kv_kernel_backend=SM86PagedKVKernelBackend(),
            ),
            replace=True,
        )
        try:
            runtime.attend(
                (state,), 0, query_decode, (1,), phase="decode"
            )
            raise AssertionError("V66 injected CUDA failure did not fire")
        except RuntimeError as exc:
            assert "injected CUDA" in str(exc)
        assert runtime.page_pool.profile()["total_pin_count"] == 0
        runtime.registry.register(original_bundle, replace=True)
        runtime.attend((state,), 0, query_decode, (1,), phase="decode")
        runtime.quiesce(timeout_seconds=10.0)

    profile = runtime.profile_stats()
    runtime.close()
    common = {
        "seed": int(seed),
        "device": torch.cuda.get_device_name(device),
        "architecture": "sm{}{}".format(*torch.cuda.get_device_capability(device)),
        "provider": policy.attention_backend,
    }
    return {
        "V60": dict(common, max_abs_error=decode_error, latency_ms=decode_ms),
        "V61": dict(common, max_abs_error=full_error, decision=full_decision),
        "V62": dict(common, max_abs_error=chunk_error),
        "V63": dict(common, streams=2, repetitions=8),
        "V64": dict(common, allocated_drift_bytes=allocated_drift, iterations=50),
        "V65": dict(common, reserved_drift_bytes=reserved_drift, iterations=50),
        "V66": dict(common, recovered=True, final_pin_count=0),
        "V67": dict(
            common,
            decode_latency_ms=decode_ms,
            drift_loop_ms=drift_ms,
            attention_calls=profile["attention_calls"],
        ),
    }
