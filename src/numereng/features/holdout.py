# --------------------------------------------------------------------------- #
# Module docstring
# --------------------------------------------------------------------------- #
"""Frozen chronological-suffix holdout helpers for the agentic research loop.

A trailing block of eras (the "frozen holdout") is carved off the chronological
tail of an experiment's era order, separated from the search region by a purge
gap. Every metric the LLM research loop sees excludes those eras; at closeout the
selected candidate is scored on the holdout exactly once, sealed, and re-opening
is refused.

These are pure helpers plus a manifest reader: partition math, a sha256 tamper
fingerprint, spec (de)serialization, and an `EraFilter` applied at the scoring
choke point. This module imports only `store` (for the experiments path layout)
so `scoring` and `experiments` can both depend on it without a cycle.

USAGE:
    from numereng.features import holdout
    spec = holdout.build_spec(era_order=eras, holdout_n_eras=52, era_gap=4)
    frame = holdout.exclusion_filter(spec).apply(frame, era_col="era")
    active = holdout.resolve_active_spec(store_root=".numereng", experiment_id=eid)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from numereng.features.store import resolve_store_root, resolve_workspace_layout_from_store_root

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MODE = "chronological_suffix"
METADATA_KEY = "agentic_research_holdout"
_ARCHIVE_DIRNAME = "_archive"

FilterMode = Literal["exclude", "restrict"]


class HoldoutError(Exception):
    """Raised on invalid holdout partitions, tamper, or reuse of a sealed holdout."""


# --------------------------------------------------------------------------- #
# Era filter
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EraFilter:
    """A row filter over a prediction frame's era column.

    `exclude` drops the named eras (loop-visible scoring hides the holdout);
    `restrict` keeps only the named eras (one-time closeout holdout scoring).
    """

    mode: FilterMode
    eras: frozenset[str]

    def apply(self, frame: pd.DataFrame, *, era_col: str) -> pd.DataFrame:
        if era_col not in frame.columns:
            raise HoldoutError(f"holdout_era_col_missing:{era_col}")
        if not self.eras:
            if self.mode == "restrict":
                raise HoldoutError("holdout_restrict_empty")
            return frame
        era_values = frame[era_col].astype(str)
        mask = era_values.isin(self.eras)
        if self.mode == "exclude":
            mask = ~mask
        return frame.loc[mask]


# --------------------------------------------------------------------------- #
# Holdout spec
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HoldoutSpec:
    """One experiment's holdout carve-out, requested at init and frozen on first train.

    A *requested* spec carries only the knobs (`holdout_n_eras`, `era_gap`). A
    *frozen* spec additionally pins `holdout_eras`, `gap_eras`, `fingerprint`, and
    `frozen_at`; `sealed` flips true after the one-time closeout scoring.
    """

    holdout_n_eras: int
    era_gap: int
    mode: str = MODE
    holdout_eras: tuple[str, ...] | None = None
    gap_eras: tuple[str, ...] | None = None
    fingerprint: str | None = None
    frozen_at: str | None = None
    sealed: bool = False

    @property
    def is_frozen(self) -> bool:
        return self.holdout_eras is not None and self.fingerprint is not None

    def to_metadata(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode,
            "holdout_n_eras": self.holdout_n_eras,
            "era_gap": self.era_gap,
            "sealed": self.sealed,
        }
        if self.holdout_eras is not None:
            payload["holdout_eras"] = list(self.holdout_eras)
        if self.gap_eras is not None:
            payload["gap_eras"] = list(self.gap_eras)
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        if self.frozen_at is not None:
            payload["frozen_at"] = self.frozen_at
        return payload


def spec_from_metadata(payload: object) -> HoldoutSpec | None:
    """Rebuild a `HoldoutSpec` from a manifest metadata value, or None when absent/invalid."""
    if not isinstance(payload, dict):
        return None
    mode = str(payload.get("mode", MODE))
    if mode != MODE:
        raise HoldoutError(f"holdout_mode_invalid:{mode}")
    try:
        holdout_n_eras = int(payload["holdout_n_eras"])
        era_gap = int(payload["era_gap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldoutError("holdout_spec_invalid") from exc
    holdout_eras = _opt_str_tuple(payload.get("holdout_eras"))
    gap_eras = _opt_str_tuple(payload.get("gap_eras"))
    fingerprint = payload.get("fingerprint")
    frozen_at = payload.get("frozen_at")
    return HoldoutSpec(
        holdout_n_eras=holdout_n_eras,
        era_gap=era_gap,
        mode=mode,
        holdout_eras=holdout_eras,
        gap_eras=gap_eras,
        fingerprint=str(fingerprint) if isinstance(fingerprint, str) else None,
        frozen_at=str(frozen_at) if isinstance(frozen_at, str) else None,
        sealed=bool(payload.get("sealed", False)),
    )


# --------------------------------------------------------------------------- #
# Partition + fingerprint
# --------------------------------------------------------------------------- #


def _era_sort_key(era: object) -> int | str:
    """Numeric-chronological era key, mirroring training/scoring era ordering.

    Digit strings sort as integers so a width-crossing tail (e.g. "999" -> "1000")
    stays chronological; non-numeric eras fall back to string order.
    """
    text = str(era)
    if text.isdigit():
        return int(text)
    return text


def partition_eras(
    era_order: tuple[str, ...], *, holdout_n_eras: int, era_gap: int
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Split a chronological era order into (search, gap, holdout) suffix blocks."""
    eras = [str(era) for era in era_order]
    if holdout_n_eras <= 0 or era_gap < 0 or holdout_n_eras + era_gap >= len(eras):
        raise HoldoutError(f"holdout_split_invalid:eras={len(eras)}:holdout={holdout_n_eras}:gap={era_gap}")
    cut_holdout = len(eras) - holdout_n_eras
    cut_gap = cut_holdout - era_gap
    search = tuple(eras[:cut_gap])
    gap = tuple(eras[cut_gap:cut_holdout])
    holdout = tuple(eras[cut_holdout:])
    return search, gap, holdout


def holdout_fingerprint(*, era_order: tuple[str, ...], holdout_eras: tuple[str, ...]) -> str:
    """sha256 over the full era order plus the holdout block; changes on any tamper."""
    payload = {"eras": [str(era) for era in era_order], "holdout": [str(era) for era in holdout_eras]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_spec(
    *, era_order: tuple[str, ...], holdout_n_eras: int, era_gap: int, frozen_at: str | None = None
) -> HoldoutSpec:
    """Partition an era order and return a frozen `HoldoutSpec` with a tamper fingerprint."""
    _, gap, holdout = partition_eras(era_order, holdout_n_eras=holdout_n_eras, era_gap=era_gap)
    fingerprint = holdout_fingerprint(era_order=era_order, holdout_eras=holdout)
    return HoldoutSpec(
        holdout_n_eras=holdout_n_eras,
        era_gap=era_gap,
        holdout_eras=holdout,
        gap_eras=gap,
        fingerprint=fingerprint,
        frozen_at=frozen_at or _utc_now_iso(),
        sealed=False,
    )


def verify_fingerprint(spec: HoldoutSpec, *, era_order: tuple[str, ...]) -> None:
    """Raise `HoldoutError` when the current era order no longer matches the frozen fingerprint."""
    if not spec.is_frozen or spec.holdout_eras is None:
        raise HoldoutError("holdout_not_frozen")
    recomputed = holdout_fingerprint(era_order=era_order, holdout_eras=spec.holdout_eras)
    if recomputed != spec.fingerprint:
        raise HoldoutError("holdout_frozen_input_tampered")


def seal(spec: HoldoutSpec) -> HoldoutSpec:
    """Return a sealed copy of a frozen spec; refuse if already sealed."""
    if not spec.is_frozen:
        raise HoldoutError("holdout_not_frozen")
    if spec.sealed:
        raise HoldoutError("holdout_reuse_blocked")
    return replace(spec, sealed=True)


# --------------------------------------------------------------------------- #
# Era filters
# --------------------------------------------------------------------------- #


def exclusion_filter(spec: HoldoutSpec) -> EraFilter | None:
    """Filter that drops holdout + gap eras from loop-visible scoring; None until frozen."""
    if not spec.is_frozen or spec.holdout_eras is None:
        return None
    eras = set(spec.holdout_eras)
    eras.update(spec.gap_eras or ())
    return EraFilter(mode="exclude", eras=frozenset(eras))


def restriction_filter(spec: HoldoutSpec) -> EraFilter:
    """Filter that keeps only the holdout eras for one-time closeout scoring."""
    if not spec.is_frozen or spec.holdout_eras is None:
        raise HoldoutError("holdout_not_frozen")
    return EraFilter(mode="restrict", eras=frozenset(spec.holdout_eras))


# --------------------------------------------------------------------------- #
# Manifest reader
# --------------------------------------------------------------------------- #


def resolve_active_spec(*, store_root: str | Path, experiment_id: str) -> HoldoutSpec | None:
    """Read one experiment's holdout spec from its manifest, or None when unset.

    Reads `experiment.json` directly (live then archived) so `scoring` can resolve
    the spec without importing `experiments` and creating an import cycle.
    """
    manifest_path = _manifest_path(store_root=store_root, experiment_id=experiment_id)
    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return spec_from_metadata(metadata.get(METADATA_KEY))


def read_prediction_era_order(predictions_path: str | Path, *, era_col: str = "era") -> tuple[str, ...]:
    """Return the numeric-chronological distinct era order from a predictions parquet."""
    frame = pd.read_parquet(Path(predictions_path), columns=[era_col])
    return tuple(sorted({str(era) for era in frame[era_col].tolist()}, key=_era_sort_key))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _manifest_path(*, store_root: str | Path, experiment_id: str) -> Path | None:
    layout = resolve_workspace_layout_from_store_root(resolve_store_root(store_root))
    live = layout.experiments_root / experiment_id / "experiment.json"
    if live.is_file():
        return live
    archived = layout.experiments_root / _ARCHIVE_DIRNAME / experiment_id / "experiment.json"
    if archived.is_file():
        return archived
    return None


def _opt_str_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise HoldoutError("holdout_spec_invalid")
    return tuple(str(item) for item in value)


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


__all__ = [
    "MODE",
    "METADATA_KEY",
    "EraFilter",
    "HoldoutError",
    "HoldoutSpec",
    "build_spec",
    "exclusion_filter",
    "holdout_fingerprint",
    "partition_eras",
    "read_prediction_era_order",
    "resolve_active_spec",
    "restriction_filter",
    "seal",
    "spec_from_metadata",
    "verify_fingerprint",
]
