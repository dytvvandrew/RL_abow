"""Finite-horizon IQL with discrete actions and small regularized models."""

from dataclasses import astuple, dataclass

import numpy as np

from .data import RELEASE_IDS, SWING_IDS, ReleaseState, SwingState, swing_levels
from .reward import score


SWING_SCALE_FLOOR = (0.01, 0.05)
RELEASE_SCALE_FLOOR = (0.02, 0.01, 0.05, 0.10, 0.5, 1.0, 1.0)
SUPPORT_RADIUS = 4.0


class UnsupportedStateError(ValueError):
    """There is no nearby recorded example for an available action."""


def _swing_features(state: SwingState):
    return astuple(state)


def _release_features(state: ReleaseState, swing_id: int):
    return (*astuple(state), *swing_levels(swing_id))


def _features(states):
    left, right = np.triu_indices(states.shape[1])
    quadratic = states[:, left] * states[:, right]
    return np.column_stack((np.ones(len(states)), states, quadratic))


def _ridge(features, targets, weights, regularization):
    weighted = features.T * weights
    penalty = np.eye(features.shape[1]) * regularization * weights.sum()
    penalty[0, 0] = 0.0
    return np.linalg.solve(weighted @ features + penalty, weighted @ targets)


def _expectile(features, targets, expectile):
    weights = np.ones(len(targets))
    for _ in range(30):
        coefficients = _ridge(features, targets, weights, regularization=0.1)
        updated = np.where(targets > features @ coefficients, expectile, 1.0 - expectile)
        if np.array_equal(updated, weights):
            break
        weights = updated
    return coefficients


def _weighted_actor(features, actions, weights, action_count):
    labels = np.eye(action_count)[actions]
    weights = weights / weights.sum()
    coefficients = np.zeros((features.shape[1], action_count))
    regularization = 0.025
    weighted_features = features.T * weights
    smoothness = 0.5 * np.linalg.norm(features * np.sqrt(weights[:, None]), ord=2) ** 2
    step_size = 1.0 / (smoothness + regularization)
    for _ in range(400):
        logits = features @ coefficients
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = weighted_features @ (probabilities - labels)
        gradient[1:] += regularization * coefficients[1:]
        coefficients -= step_size * gradient
    return coefficients


@dataclass
class Stage:
    center: np.ndarray
    scale: np.ndarray
    critic: np.ndarray
    value: np.ndarray
    actor: np.ndarray
    support_states: np.ndarray
    support_actions: np.ndarray
    support_groups: np.ndarray

    def standardize(self, states):
        return (np.atleast_2d(states) - self.center) / self.scale

    def state_values(self, states):
        return _features(self.standardize(states)) @ self.value

    def observed_action_value(self, state, action):
        if action not in self.support_actions:
            return None
        return float((_features(self.standardize(state)) @ self.critic[:, action])[0])

    def choose(self, state, group=0, available=None):
        standardized = self.standardize(state)
        distances = np.linalg.norm(self.support_states - standardized, axis=1)
        supported = np.unique(self.support_actions[
            (self.support_groups == group) & (distances <= SUPPORT_RADIUS)
        ])
        if available is not None:
            supported = np.intersect1d(supported, available)
        if not len(supported):
            raise UnsupportedStateError("no available action has nearby training data")
        logits = (_features(standardized) @ self.actor)[0]
        return int(supported[np.argmax(logits[supported])])

    def to_dict(self):
        return {name: values.tolist() for name, values in vars(self).items()}

    @classmethod
    def from_dict(cls, value, state_width, action_count, group_count):
        feature_width = 1 + state_width + state_width * (state_width + 1) // 2
        shapes = {
            "center": (state_width,),
            "scale": (state_width,),
            "critic": (feature_width, action_count),
            "value": (feature_width,),
            "actor": (feature_width, action_count),
        }
        arrays = {name: np.asarray(value[name], dtype=float) for name in shapes}
        for name, shape in shapes.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"invalid model field {name}")
        if (arrays["scale"] <= 0).any():
            raise ValueError("model scales must be positive")
        states = np.asarray(value["support_states"], dtype=float)
        actions = np.asarray(value["support_actions"])
        groups = np.asarray(value["support_groups"])
        if (
            actions.ndim != 1 or not len(actions) or groups.shape != actions.shape
            or states.shape != (len(actions), state_width) or not np.isfinite(states).all()
            or actions.dtype.kind not in "iu" or not np.all((0 <= actions) & (actions < action_count))
            or groups.dtype.kind not in "iu" or not np.all((0 <= groups) & (groups < group_count))
        ):
            raise ValueError("invalid model support data")
        return cls(**arrays, support_states=states, support_actions=actions, support_groups=groups)


def _fit_stage(states, actions, targets, groups, scale_floor, action_count, expectile, beta):
    states = np.asarray(states)
    actions = np.asarray(actions, dtype=int)
    targets = np.asarray(targets)
    center = states.mean(axis=0)
    scale = np.maximum(states.std(axis=0), scale_floor)
    standardized = (states - center) / scale
    features = _features(standardized)

    critic = np.zeros((features.shape[1], action_count))
    for action in np.unique(actions):
        rows = actions == action
        critic[:, action] = _ridge(features[rows], targets[rows], np.ones(rows.sum()), 0.2)
    logged_values = np.sum(features * critic[:, actions].T, axis=1)
    value = _expectile(features, logged_values, expectile)
    advantage = logged_values - features @ value
    weights = np.exp(np.clip(beta * advantage, -8.0, 3.0))
    actor = _weighted_actor(features, actions, weights, action_count)
    return Stage(center, scale, critic, value, actor, standardized, actions, np.asarray(groups))


@dataclass
class TwoStageIQL:
    swing: Stage
    release: Stage

    def choose_swing(self, state, available=None):
        return self.swing.choose(_swing_features(state), available=available)

    def choose_release(self, state, swing_id, available=None):
        return self.release.choose(_release_features(state, swing_id), swing_id, available)

    def logged_release_value(self, state, swing_id, release_id):
        return self.release.observed_action_value(_release_features(state, swing_id), release_id)

    def to_dict(self):
        return {"swing": self.swing.to_dict(), "release": self.release.to_dict()}

    @classmethod
    def from_dict(cls, value):
        return cls(
            Stage.from_dict(value["swing"], len(SWING_SCALE_FLOOR), len(SWING_IDS), 1),
            Stage.from_dict(value["release"], len(RELEASE_SCALE_FLOOR), len(RELEASE_IDS), len(SWING_IDS)),
        )


def fit_iql(trials, expectile=0.75, beta=5.0):
    """Fit terminal Q first, then back up V to the first decision with gamma=1."""
    if not 0.5 <= expectile < 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("expectile must be in [0.5, 1) and beta must be finite and nonnegative")
    labeled = [(trial, score(trial.outcome)) for trial in trials]
    labeled = [(trial, result) for trial, result in labeled if result is not None]
    completed = [(trial, result) for trial, result in labeled if trial.release_id is not None]
    if not completed:
        raise ValueError("training requires scored trials with a release decision")

    release_states = [_release_features(trial.release_state, trial.swing_id) for trial, _ in completed]
    release = _fit_stage(
        release_states,
        [trial.release_id for trial, _ in completed],
        [result.reward for _, result in completed],
        [trial.swing_id for trial, _ in completed],
        RELEASE_SCALE_FLOOR, len(RELEASE_IDS), expectile, beta,
    )
    # A trial that terminated before the second decision has no continuation value.
    continuation = dict(zip(
        (trial.trial_id for trial, _ in completed),
        release.state_values(release_states),
    ))
    swing = _fit_stage(
        [_swing_features(trial.swing_state) for trial, _ in labeled],
        [trial.swing_id for trial, _ in labeled],
        [continuation.get(trial.trial_id, result.reward) for trial, result in labeled],
        [0] * len(labeled),
        SWING_SCALE_FLOOR, len(SWING_IDS), expectile, beta,
    )
    return TwoStageIQL(swing, release)
