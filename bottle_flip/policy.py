"""Load trained policies once for repeated inference."""

import json
from pathlib import Path

from .baseline import FixedPolicy
from .data import RELEASE_IDS, SWING_IDS, action_id
from .iql import TwoStageIQL


MODEL_VERSION = 1


def load_policies(path):
    with Path(path).open(encoding="utf-8") as stream:
        bundle = json.load(stream)
    if (
        not isinstance(bundle, dict) or type(bundle.get("model_version")) is not int
        or bundle["model_version"] != MODEL_VERSION
    ):
        raise ValueError(f"invalid model bundle; expected model_version {MODEL_VERSION}")
    fixed = bundle["fixed"]
    return {
        "fixed": FixedPolicy(
            action_id(fixed["swing_id"], "swing_id", SWING_IDS),
            action_id(fixed["release_id"], "release_id", RELEASE_IDS),
        ),
        "iql": TwoStageIQL.from_dict(bundle["iql"]),
    }
