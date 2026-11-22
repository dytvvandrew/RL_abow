"""Offline labels shared by training and closed-loop evaluation."""

from dataclasses import dataclass

from .data import Outcome


@dataclass(frozen=True)
class Score:
    reward: float
    success: bool
    result: str


def score(outcome: Outcome) -> Score | None:
    if outcome.status == "unobserved":
        return None
    if outcome.status != "landed":
        return Score(0.0, False, outcome.status)
    if not outcome.in_target:
        return Score(0.0, False, "outside_target")

    magnitude = abs(outcome.flip_deg)
    turn_error = abs(magnitude - 360)
    if magnitude < 340:
        result = "under_rotated"
    elif magnitude > 380:
        result = "over_rotated"
    elif not outcome.bottom_first:
        result = "not_bottom_first"
    elif outcome.upright_s < 2.0:
        result = "did_not_stay_upright"
    else:
        return Score(1.0, True, "success")
    rotation = 0.10 * max(0.0, 1.0 - turn_error / 90.0)
    tilt = 0.05 * max(0.0, 1.0 - outcome.impact_tilt_deg / 45.0)
    return Score(rotation + tilt, False, result)
