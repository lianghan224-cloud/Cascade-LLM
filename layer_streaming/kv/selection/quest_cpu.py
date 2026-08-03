"""Deterministic CPU reference index for Quest-style page selection.

Normal queries use compact per-dimension extrema. Raw rows are retained only
for transactional append/rollback and debug validation; budget queries never
rescan the original keys.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
import uuid

from ..errors import KVLifecycleError


QUEST_INDEX_FORMAT_VERSION = 1


@dataclass(frozen=True, order=True)
class LogicalKVBlockId:
    model_id: str
    session_id: str
    branch_id: str
    layer: int
    logical_block: int

    def key(self):
        return "{}|{}|{}|{}|{}".format(
            self.model_id,
            self.session_id,
            self.branch_id,
            int(self.layer),
            int(self.logical_block),
        )

    @classmethod
    def from_key(cls, value):
        model, session, branch, layer, logical = str(value).split("|", 4)
        return cls(model, session, branch, int(layer), int(logical))


@dataclass(frozen=True)
class IndexRecordId:
    value: str

    @classmethod
    def create(cls, logical_block_id, data_version):
        digest = hashlib.sha256(
            "{}@{}@{}".format(
                logical_block_id.key(), int(data_version), uuid.uuid4().hex
            ).encode("utf-8")
        ).hexdigest()
        return cls(digest)


@dataclass(frozen=True)
class QuestIndexRecord:
    record_id: IndexRecordId
    logical_block_id: LogicalKVBlockId
    token_start: int
    token_count: int
    data_version: int
    index_version: int
    dimensions: int
    minimum: tuple
    maximum: tuple
    mean: tuple
    rows: tuple
    checksum: str
    format_version: int = QUEST_INDEX_FORMAT_VERSION

    def as_dict(self, include_rows=True):
        result = {
            "record_id": self.record_id.value,
            "logical_block_id": self.logical_block_id.key(),
            "token_start": int(self.token_start),
            "token_count": int(self.token_count),
            "data_version": int(self.data_version),
            "index_version": int(self.index_version),
            "dimensions": int(self.dimensions),
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
            "mean": list(self.mean),
            "checksum": self.checksum,
            "format_version": int(self.format_version),
        }
        if include_rows:
            result["rows"] = [list(item) for item in self.rows]
        return result


@dataclass(frozen=True)
class QuestSelectionResult:
    records: tuple
    mode: str
    candidate_count: int
    selected_count: int
    budget: int
    elapsed_ms: float
    scores: tuple
    recall: object = None
    max_score_error: object = None

    @property
    def logical_block_ids(self):
        return tuple(item.logical_block_id for item in self.records)

    def as_dict(self):
        return {
            "mode": self.mode,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "budget": self.budget,
            "elapsed_ms": self.elapsed_ms,
            "scores": list(self.scores),
            "recall": self.recall,
            "max_score_error": self.max_score_error,
            "record_ids": [item.record_id.value for item in self.records],
        }


def _to_rows(block_data):
    if hasattr(block_data, "detach"):
        value = block_data.detach().float().cpu()
        if value.ndim == 1:
            value = value.reshape(1, -1)
        elif value.ndim > 2:
            value = value.reshape(-1, value.shape[-1])
        block_data = value.tolist()
    block_data = list(block_data)
    if not block_data:
        return ()
    if isinstance(block_data[0], (int, float)):
        block_data = (block_data,)
    rows = tuple(tuple(float(item) for item in row) for row in block_data)
    dimensions = len(rows[0])
    if dimensions <= 0 or any(len(row) != dimensions for row in rows):
        raise ValueError("Quest index rows must have one stable dimension")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("Quest index rows must contain finite values")
    return rows


def _checksum(rows, metadata):
    payload = json.dumps(
        {"rows": rows, "metadata": metadata},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class QuestCPUIndex:
    """Version-checked, serializable page-summary index."""

    def __init__(self):
        self._records = {}
        self._references = {}
        self.query_count = 0
        self.candidate_count = 0
        self.selected_count = 0
        self.query_ms = 0.0

    @staticmethod
    def _summaries(rows):
        if not rows:
            return (), (), ()
        columns = tuple(zip(*rows))
        return (
            tuple(min(column) for column in columns),
            tuple(max(column) for column in columns),
            tuple(sum(column) / float(len(column)) for column in columns),
        )

    def build(self, block_data, metadata):
        metadata = dict(metadata)
        logical = metadata.get("logical_block_id")
        if not isinstance(logical, LogicalKVBlockId):
            raise TypeError("logical_block_id must be LogicalKVBlockId")
        data_version = int(metadata["data_version"])
        token_start = int(metadata.get("token_start", 0))
        rows = _to_rows(block_data)
        minimum, maximum, mean = self._summaries(rows)
        record = QuestIndexRecord(
            record_id=IndexRecordId.create(logical, data_version),
            logical_block_id=logical,
            token_start=token_start,
            token_count=len(rows),
            data_version=data_version,
            index_version=data_version,
            dimensions=(len(rows[0]) if rows else 0),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            rows=rows,
            checksum=_checksum(
                rows,
                {
                    "logical_block_id": logical.key(),
                    "token_start": token_start,
                    "data_version": data_version,
                },
            ),
        )
        self._records[record.record_id.value] = record
        self._references[record.record_id.value] = 1
        return record

    def update_append(self, index_record, appended_data, new_version):
        self._require_current(index_record)
        rows = index_record.rows + _to_rows(appended_data)
        return self.build(
            rows,
            {
                "logical_block_id": index_record.logical_block_id,
                "token_start": index_record.token_start,
                "data_version": int(new_version),
            },
        )

    @staticmethod
    def _score(query, record):
        query = tuple(float(item) for item in query)
        if len(query) != record.dimensions:
            raise ValueError(
                "query dimension {} does not match index dimension {}".format(
                    len(query), record.dimensions
                )
            )
        return sum(
            max(value * lower, value * upper)
            for value, lower, upper in zip(
                query, record.minimum, record.maximum
            )
        )

    def select(self, query, candidates, budget=0, mode="full", exact_scores=None):
        started = time.perf_counter()
        candidates = tuple(candidates)
        for record in candidates:
            self._require_current(record)
            if record.index_version != record.data_version:
                raise KVLifecycleError(
                    "stale Quest index {}: index_version={} data_version={}".format(
                        record.record_id.value,
                        record.index_version,
                        record.data_version,
                    )
                )
        ordered = tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.token_start,
                    item.logical_block_id.logical_block,
                    item.record_id.value,
                ),
            )
        )
        mode = str(mode).lower().replace("topk", "budget")
        score_pairs = tuple((record, self._score(query, record)) for record in ordered)
        if mode == "full":
            selected = ordered
            resolved_budget = len(ordered)
        elif mode in {"budget", "debug_exact"}:
            resolved_budget = int(budget)
            if resolved_budget < 0:
                raise ValueError("selection budget must not be negative")
            resolved_budget = min(resolved_budget, len(ordered))
            if mode == "debug_exact" and exact_scores is None:
                raise ValueError("debug_exact requires exact_scores")
            ranked = sorted(
                score_pairs,
                key=lambda pair: (
                    -float(
                        exact_scores.get(pair[0].record_id.value, pair[1])
                        if exact_scores is not None and mode == "debug_exact"
                        else pair[1]
                    ),
                    pair[0].token_start,
                    pair[0].record_id.value,
                ),
            )[:resolved_budget]
            selected_ids = {item.record_id.value for item, _ in ranked}
            selected = tuple(item for item in ordered if item.record_id.value in selected_ids)
        else:
            raise ValueError("unknown Quest selection mode {!r}".format(mode))
        recall = None
        max_score_error = None
        if exact_scores is not None:
            exact_ranked = sorted(
                ordered,
                key=lambda item: (
                    -float(exact_scores.get(item.record_id.value, -float("inf"))),
                    item.token_start,
                ),
            )[:len(selected)]
            exact_ids = {item.record_id.value for item in exact_ranked}
            selected_ids = {item.record_id.value for item in selected}
            recall = 1.0 if not exact_ids else len(exact_ids & selected_ids) / float(len(exact_ids))
            max_score_error = max(
                (
                    abs(score - float(exact_scores.get(record.record_id.value, score)))
                    for record, score in score_pairs
                ),
                default=0.0,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.query_count += 1
        self.candidate_count += len(ordered)
        self.selected_count += len(selected)
        self.query_ms += elapsed_ms
        selected_score = {record.record_id.value: score for record, score in score_pairs}
        return QuestSelectionResult(
            records=selected,
            mode=mode,
            candidate_count=len(ordered),
            selected_count=len(selected),
            budget=resolved_budget,
            elapsed_ms=elapsed_ms,
            scores=tuple(selected_score[item.record_id.value] for item in selected),
            recall=recall,
            max_score_error=max_score_error,
        )

    def fork_ref(self, index_record):
        self._require_current(index_record)
        key = index_record.record_id.value
        self._references[key] += 1
        return index_record

    def cow_clone(self, index_record, logical_block_id=None):
        self._require_current(index_record)
        return self.build(
            index_record.rows,
            {
                "logical_block_id": logical_block_id or index_record.logical_block_id,
                "token_start": index_record.token_start,
                "data_version": index_record.data_version,
            },
        )

    def rollback(self, index_record, target_token_count, target_version):
        self._require_current(index_record)
        target_token_count = int(target_token_count)
        if target_token_count < 0 or target_token_count > index_record.token_count:
            raise ValueError("invalid Quest rollback token count")
        return self.build(
            index_record.rows[:target_token_count],
            {
                "logical_block_id": index_record.logical_block_id,
                "token_start": index_record.token_start,
                "data_version": int(target_version),
            },
        )

    def serialize(self, index_record):
        self._require_current(index_record)
        return json.dumps(
            index_record.as_dict(include_rows=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def deserialize(self, payload):
        value = json.loads(bytes(payload).decode("utf-8"))
        if int(value["format_version"]) != QUEST_INDEX_FORMAT_VERSION:
            raise ValueError("unsupported Quest index format")
        record = QuestIndexRecord(
            record_id=IndexRecordId(str(value["record_id"])),
            logical_block_id=LogicalKVBlockId.from_key(value["logical_block_id"]),
            token_start=int(value["token_start"]),
            token_count=int(value["token_count"]),
            data_version=int(value["data_version"]),
            index_version=int(value["index_version"]),
            dimensions=int(value["dimensions"]),
            minimum=tuple(float(item) for item in value["minimum"]),
            maximum=tuple(float(item) for item in value["maximum"]),
            mean=tuple(float(item) for item in value["mean"]),
            rows=tuple(tuple(float(item) for item in row) for row in value["rows"]),
            checksum=str(value["checksum"]),
            format_version=int(value["format_version"]),
        )
        expected = _checksum(
            record.rows,
            {
                "logical_block_id": record.logical_block_id.key(),
                "token_start": record.token_start,
                "data_version": record.data_version,
            },
        )
        if record.checksum != expected or record.token_count != len(record.rows):
            raise ValueError("Quest index checksum or length mismatch")
        self._records[record.record_id.value] = record
        self._references.setdefault(record.record_id.value, 1)
        return record

    def validate(self, index_record, page_metadata):
        self._require_current(index_record)
        metadata = dict(page_metadata)
        expected_logical = metadata.get("logical_block_id")
        if expected_logical is not None and expected_logical != index_record.logical_block_id:
            raise KVLifecycleError("Quest logical block identity mismatch")
        if int(metadata["data_version"]) != index_record.data_version:
            raise KVLifecycleError(
                "Quest index/page version mismatch: {} != {}".format(
                    index_record.index_version, metadata["data_version"]
                )
            )
        if int(metadata.get("token_count", index_record.token_count)) != index_record.token_count:
            raise KVLifecycleError("Quest index/page token count mismatch")
        if index_record.index_version != index_record.data_version:
            raise KVLifecycleError("Quest index is not queryable while stale")
        return True

    def invalidate_for_test(self, index_record, index_version):
        return replace(index_record, index_version=int(index_version))

    def _require_current(self, index_record):
        if not isinstance(index_record, QuestIndexRecord):
            raise TypeError("expected QuestIndexRecord")
        current = self._records.get(index_record.record_id.value)
        if current is not index_record and current != index_record:
            raise KVLifecycleError("unknown Quest index record")

    def stats(self):
        return {
            "records": len(self._records),
            "queries": self.query_count,
            "candidates": self.candidate_count,
            "selected": self.selected_count,
            "query_ms": self.query_ms,
            "selection_ratio": (
                self.selected_count / float(self.candidate_count)
                if self.candidate_count
                else 1.0
            ),
        }
