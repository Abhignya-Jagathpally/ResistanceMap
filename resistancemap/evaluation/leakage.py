"""Leakage detection utilities for evaluation governance.

Two narrowly-scoped audits live here:

* :func:`audit_patient_split` -- structural overlap of patient IDs across
  train / val / test splits.
* :func:`detect_temporal_leakage` -- per-patient time ordering across splits,
  i.e. ``max(train_time(pid)) >= min(test_time(pid))``.

Both functions are pure-python and treat numpy as optional. They never
fabricate data: if the inputs are empty or the timestamp field is missing,
they record that explicitly in the returned :class:`LeakageReport`.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LeakageReport:
    """Result of a leakage audit.

    Attributes:
        duplicate_patient_ids: Patient IDs that appear in more than one
            split. The dict maps ``patient_id -> [split_names]``.
        temporal_leaks: List of dicts describing patients whose train-time
            is greater than or equal to their test-time. Each entry has
            ``patient_id``, ``train_time``, ``test_time``, ``split_pair``.
        clean: ``True`` iff there were no duplicates **and** no temporal
            leaks **and** the audit had something to audit.
        details: Free-form bag for additional diagnostics
            (e.g. ``audited_n``, ``timestamp_field``, ``warnings``).
    """

    duplicate_patient_ids: dict[str, list[str]] = field(default_factory=dict)
    temporal_leaks: list[dict[str, Any]] = field(default_factory=list)
    clean: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def overlaps(self) -> list[str]:
        """Patient IDs that appear in more than one split."""
        return list(self.duplicate_patient_ids.keys())

    @property
    def violations(self) -> list[dict[str, Any]]:
        """Temporal leakage violations (alias for :attr:`temporal_leaks`)."""
        return list(self.temporal_leaks)

    @property
    def leaking_patients(self) -> list[str]:
        """Patient IDs flagged for temporal leakage."""
        return [v.get("patient_id") for v in self.temporal_leaks if "patient_id" in v]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view of the report."""
        return {
            "duplicate_patient_ids": dict(self.duplicate_patient_ids),
            "temporal_leaks": [dict(t) for t in self.temporal_leaks],
            "clean": self.clean,
            "details": dict(self.details),
        }


def _to_id_list(ids: Iterable[Any]) -> list[str]:
    """Coerce an iterable of IDs into a list of strings (preserving order)."""
    out: list[str] = []
    for item in ids:
        if item is None:
            continue
        out.append(str(item))
    return out


def audit_patient_split(
    train_ids: Iterable[Any] | None = None,
    val_ids: Iterable[Any] | None = None,
    test_ids: Iterable[Any] | None = None,
    *,
    train: Iterable[Any] | None = None,
    val: Iterable[Any] | None = None,
    test: Iterable[Any] | None = None,
) -> LeakageReport:
    """Detect duplicate patient IDs across train / val / test splits.

    Args:
        train_ids / train: Iterable of patient IDs assigned to the training
            split. Either name is accepted; ``train`` is the canonical
            short form for callers that prefer keyword arguments.
        val_ids / val: Iterable of patient IDs assigned to the validation
            split.
        test_ids / test: Iterable of patient IDs assigned to the test split.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` only when none of
        the inputs were empty and there were zero overlaps.
    """
    train_arg = train_ids if train_ids is not None else (train if train is not None else [])
    val_arg = val_ids if val_ids is not None else (val if val is not None else [])
    test_arg = test_ids if test_ids is not None else (test if test is not None else [])
    train = _to_id_list(train_arg)
    val = _to_id_list(val_arg)
    test = _to_id_list(test_arg)

    membership: dict[str, list[str]] = defaultdict(list)
    for pid in train:
        membership[pid].append("train")
    for pid in val:
        membership[pid].append("val")
    for pid in test:
        membership[pid].append("test")

    duplicates = {
        pid: splits for pid, splits in membership.items() if len(set(splits)) > 1
    }

    details: dict[str, Any] = {
        "n_train": len(set(train)),
        "n_val": len(set(val)),
        "n_test": len(set(test)),
        "n_unique_total": len(membership),
    }
    warnings: list[str] = []
    if not train or not test:
        warnings.append("train or test split is empty; audit is vacuous")
    if not val:
        warnings.append("validation split is empty")
    if warnings:
        details["warnings"] = warnings

    clean = not duplicates and bool(train) and bool(test)
    if duplicates:
        logger.error(
            "audit_patient_split: %d patient(s) overlap across splits",
            len(duplicates),
        )
    return LeakageReport(
        duplicate_patient_ids=duplicates,
        temporal_leaks=[],
        clean=clean,
        details=details,
    )


def _coerce_timestamp(value: Any) -> float | None:
    """Best-effort conversion of an arbitrary timestamp into a float.

    Accepts ``datetime``, ``date``, ISO-format strings, or numeric values.
    Returns ``None`` if the value cannot be parsed (the caller decides what
    to do; this function never raises on bad input).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day).timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def detect_temporal_leakage(
    samples: Sequence[Mapping[str, Any]] | None = None,
    split_assignments: Mapping[Any, str] | None = None,
    timestamp_field: str = "collection_date",
    patient_id_field: str = "patient_id",
    *,
    train: Sequence[Mapping[str, Any]] | None = None,
    val: Sequence[Mapping[str, Any]] | None = None,
    test: Sequence[Mapping[str, Any]] | None = None,
) -> LeakageReport:
    """Flag temporal leakage where train samples are not strictly before test.

    For every patient that appears in more than one split, the function
    compares ``max(train_time)`` against ``min(test_time)`` (and val against
    test, and train against val). If the train-side max is greater than or
    equal to the test-side min, the patient is flagged.

    Args:
        samples: Iterable of sample dicts. Each dict must carry at least
            ``patient_id_field`` and ``timestamp_field``.
        split_assignments: Mapping ``sample_index -> split_name`` (split
            name in ``{"train", "val", "test"}``). Sample index matches the
            position in ``samples``.
        timestamp_field: Key inside each sample dict carrying the
            collection timestamp. Defaults to ``"collection_date"``.
        patient_id_field: Key inside each sample dict carrying the patient
            identifier. Defaults to ``"patient_id"``.

    Returns:
        A :class:`LeakageReport`. ``clean`` is ``True`` iff no leaks were
        found *and* the audit had usable timestamps to inspect.
    """
    # Polymorphic entry: if the caller passed `train=`/`val=`/`test=`,
    # synthesise a flat sample list and split_assignments. The "timestamp"
    # field is also accepted as an alias for ``timestamp_field``.
    if samples is None and (train is not None or val is not None or test is not None):
        flat: list[Mapping[str, Any]] = []
        assignments: dict[int, str] = {}
        for split_name, split_samples in (("train", train), ("val", val), ("test", test)):
            if not split_samples:
                continue
            for s in split_samples:
                assignments[len(flat)] = split_name
                flat.append(s)
        samples = flat
        split_assignments = assignments
        # Detect short alias "timestamp" if the caller used that and the
        # default ``timestamp_field`` does not occur on the records.
        if flat and timestamp_field not in flat[0] and "timestamp" in flat[0]:
            timestamp_field = "timestamp"

    if samples is None or split_assignments is None:
        raise TypeError(
            "detect_temporal_leakage requires either (samples, split_assignments) "
            "or one of train=/val=/test= sample lists"
        )

    by_patient_split: dict[tuple[str, str], list[float]] = defaultdict(list)
    parse_failures = 0
    missing_field = 0

    for idx, sample in enumerate(samples):
        split = split_assignments.get(idx)
        if split is None:
            continue
        if patient_id_field not in sample or timestamp_field not in sample:
            missing_field += 1
            continue
        pid = str(sample[patient_id_field])
        ts = _coerce_timestamp(sample[timestamp_field])
        if ts is None:
            parse_failures += 1
            continue
        by_patient_split[(pid, split)].append(ts)

    leaks: list[dict[str, Any]] = []
    patients = {pid for (pid, _) in by_patient_split}
    pairs = (("train", "test"), ("train", "val"), ("val", "test"))

    for pid in patients:
        for early, late in pairs:
            early_times = by_patient_split.get((pid, early))
            late_times = by_patient_split.get((pid, late))
            if not early_times or not late_times:
                continue
            if max(early_times) >= min(late_times):
                leaks.append(
                    {
                        "patient_id": pid,
                        "train_time": max(early_times),
                        "test_time": min(late_times),
                        "split_pair": f"{early}->{late}",
                    }
                )

    details: dict[str, Any] = {
        "audited_n": len(samples),
        "timestamp_field": timestamp_field,
        "patient_id_field": patient_id_field,
        "parse_failures": parse_failures,
        "missing_field_count": missing_field,
        "patients_with_multi_split": sum(
            1
            for pid in patients
            if sum(
                1
                for s in ("train", "val", "test")
                if (pid, s) in by_patient_split
            )
            > 1
        ),
    }
    warnings: list[str] = []
    if not samples:
        warnings.append("no samples provided; temporal audit is vacuous")
    if missing_field:
        warnings.append(
            f"{missing_field} sample(s) missing {patient_id_field!r} or "
            f"{timestamp_field!r}; not audited"
        )
    if parse_failures:
        warnings.append(
            f"{parse_failures} timestamp(s) could not be parsed; not audited"
        )
    if warnings:
        details["warnings"] = warnings

    clean = not leaks and bool(samples) and not missing_field and not parse_failures
    if leaks:
        logger.error(
            "detect_temporal_leakage: %d patient/split pair(s) leak", len(leaks)
        )
    return LeakageReport(
        duplicate_patient_ids={},
        temporal_leaks=leaks,
        clean=clean,
        details=details,
    )
