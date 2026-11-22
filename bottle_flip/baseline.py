"""Fit one fixed swing-and-release combination from the training sessions."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from .iql import UnsupportedStateError
from .reward import score


@dataclass(frozen=True)
class FixedPolicy:
    swing_id: int
    release_id: int

    def choose_swing(self, state, available=None):
        if available is not None and self.swing_id not in available:
            raise UnsupportedStateError("the fixed swing is unavailable")
        return self.swing_id

    def choose_release(self, state, swing_id, available=None):
        if swing_id != self.swing_id or (available is not None and self.release_id not in available):
            raise UnsupportedStateError("the fixed release is unavailable for this swing")
        return self.release_id


def fit_baseline(trials):
    attempts = Counter()
    reached_release = Counter()
    pairs = defaultdict(list)
    for trial in trials:
        result = score(trial.outcome)
        if result is None:
            continue
        attempts[trial.swing_id] += 1
        if trial.release_id is not None:
            reached_release[trial.swing_id] += 1
            pairs[trial.swing_id, trial.release_id].append(result)
    if not pairs:
        raise ValueError("the fixed baseline needs a scored trial with a release decision")

    def ranking(pair):
        results = pairs[pair]
        swing = pair[0]
        reach_rate = reached_release[swing] / attempts[swing]
        success = reach_rate * sum(result.success for result in results) / len(results)
        reward = reach_rate * sum(result.reward for result in results) / len(results)
        return success, reward, len(results), -pair[0], -pair[1]

    return FixedPolicy(*max(pairs, key=ranking))
