"""Shared, versioned trial format for simulation and real recordings."""

import json
import math
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
SWING_IDS = range(9)
RELEASE_IDS = range(3)
OUTCOME_STATUSES = {"landed", "no_release", "aborted", "unobserved"}


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def action_id(value, name, allowed):
    if type(value) is not int or value not in allowed:
        raise ValueError(f"{name} must be an integer in {list(allowed)}")
    return value


def swing_levels(swing_id):
    lift, pitch = divmod(swing_id, 3)
    return lift - 1, pitch - 1


@dataclass(frozen=True)
class SwingState:
    height_m: float
    pitch_rad: float

    @classmethod
    def from_dict(cls, value):
        value = _object(value, "swing_state")
        return cls(*(_number(value.get(f.name), f.name) for f in fields(cls)))


@dataclass(frozen=True)
class ReleaseState:
    phase: float
    height_m: float
    pitch_rad: float
    vz_m_s: float
    pitch_rate_rad_s: float

    @classmethod
    def from_dict(cls, value):
        value = _object(value, "release_state")
        state = cls(*(_number(value.get(f.name), f.name) for f in fields(cls)))
        if not 0 <= state.phase <= 1:
            raise ValueError("phase must be between 0 and 1")
        return state


@dataclass(frozen=True)
class Outcome:
    status: str
    flip_deg: float | None = None
    impact_tilt_deg: float | None = None
    bottom_first: bool | None = None
    in_target: bool | None = None
    upright_s: float | None = None

    @classmethod
    def from_dict(cls, value):
        value = _object(value, "outcome")
        status = value.get("status")
        if not isinstance(status, str) or status not in OUTCOME_STATUSES:
            raise ValueError(f"outcome.status must be one of {sorted(OUTCOME_STATUSES)}")
        if status != "landed":
            return cls(status)
        flip = _number(value.get("flip_deg"), "flip_deg")
        tilt = _number(value.get("impact_tilt_deg"), "impact_tilt_deg")
        upright = _number(value.get("upright_s"), "upright_s")
        if not 0 <= tilt <= 180 or upright < 0:
            raise ValueError("impact_tilt_deg must be in [0, 180]; upright_s must be nonnegative")
        if type(value.get("bottom_first")) is not bool or type(value.get("in_target")) is not bool:
            raise ValueError("landed outcomes require boolean bottom_first and in_target")
        return cls(status, flip, tilt, value["bottom_first"], value["in_target"], upright)


@dataclass(frozen=True)
class Trial:
    trial_id: str
    session_id: str
    swing_state: SwingState
    swing_id: int
    release_state: ReleaseState | None
    release_id: int | None
    outcome: Outcome
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        value = _object(value, "trial")
        if type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        outcome = Outcome.from_dict(value.get("outcome"))
        release_state = value.get("release_state")
        release_id = value.get("release_id")
        if (release_state is None) != (release_id is None):
            raise ValueError("release_state and release_id must both be present or both null")
        if release_id is None and outcome.status == "landed":
            raise ValueError("a landed outcome requires a release decision")
        return cls(
            _identifier(value.get("trial_id"), "trial_id"),
            _identifier(value.get("session_id"), "session_id"),
            SwingState.from_dict(value.get("swing_state")),
            action_id(value.get("swing_id"), "swing_id", SWING_IDS),
            ReleaseState.from_dict(release_state) if release_state is not None else None,
            action_id(release_id, "release_id", RELEASE_IDS) if release_id is not None else None,
            outcome,
            _object(value.get("metadata", {}), "metadata"),
        )

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def read_trials(path: Path):
    trials = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                trial = Trial.from_dict(json.loads(line))
                if trial.trial_id in seen:
                    raise ValueError(f"duplicate trial_id {trial.trial_id!r}")
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            seen.add(trial.trial_id)
            trials.append(trial)
    if not trials:
        raise ValueError(f"{path}: no trials found")
    return trials


def write_trials(stream, trials: Iterable[Trial]):
    for trial in trials:
        print(json.dumps(trial.to_dict(), allow_nan=False, ensure_ascii=False), file=stream)


def split_sessions(trials, seed):
    """Keep all trials from a recording session in the same partition."""
    sessions = sorted({trial.session_id for trial in trials})
    random.Random(seed).shuffle(sessions)
    held_out = max(1, len(sessions) // 5)
    if len(sessions) < 3:
        raise ValueError("need three sessions for a split; use --all-data to train without a holdout")
    train_end = len(sessions) - 2 * held_out
    validation_end = len(sessions) - held_out
    partitions = {
        "train": set(sessions[:train_end]),
        "validation": set(sessions[train_end:validation_end]),
        "test": set(sessions[validation_end:]),
    }
    return {
        name: [trial for trial in trials if trial.session_id in session_ids]
        for name, session_ids in partitions.items()
    }
