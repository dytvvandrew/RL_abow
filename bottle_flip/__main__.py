"""Command-line workflow for collection, offline training and inference."""

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

from .baseline import fit_baseline
from .data import (
    RELEASE_IDS, SWING_IDS, ReleaseState, SwingState, action_id, read_trials,
    split_sessions, swing_levels, write_trials,
)
from .iql import UnsupportedStateError, fit_iql
from .policy import MODEL_VERSION, load_policies
from .reward import score
from .simulator import collect_trials, simulate_trial


def _dump(value, stream):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), file=stream)


def _write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            write(stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _simulate(args):
    _write(args.output, lambda stream: write_trials(stream, collect_trials(args.sessions, args.seed)))
    return {"output": str(args.output), "sessions": args.sessions, "trials": args.sessions * 27}


def _describe(trials, iql):
    labeled = [(trial, result) for trial in trials if (result := score(trial.outcome)) is not None]
    errors = []
    for trial, result in labeled:
        if trial.release_id is not None:
            value = iql.logged_release_value(trial.release_state, trial.swing_id, trial.release_id)
            if value is not None:
                errors.append(abs(value - result.reward))
    return {
        "sessions": sorted({trial.session_id for trial in trials}),
        "trials": len(trials),
        "scored_trials": len(labeled),
        "successful_trials": sum(result.success for _, result in labeled),
        "logged_release_reward_mae": mean(errors) if errors else None,
        "release_reward_mae_samples": len(errors),
    }


def _train(args):
    if args.data.resolve() == args.output.resolve():
        raise ValueError("the model output must differ from the trial data path")
    trials = read_trials(args.data)
    partitions = {"train": trials} if args.all_data else split_sessions(trials, args.seed)
    iql = fit_iql(partitions["train"], expectile=args.expectile, beta=args.beta)
    fixed = fit_baseline(partitions["train"])
    bundle = {
        "model_version": MODEL_VERSION,
        "fixed": asdict(fixed),
        "iql": iql.to_dict(),
    }
    report = {name: _describe(rows, iql) for name, rows in partitions.items()}
    _write(args.output, lambda stream: _dump(bundle, stream))
    return {"output": str(args.output), "fixed": asdict(fixed), "partitions": report}


def _infer(args):
    policy = load_policies(args.model)[args.policy]
    if args.input is None:
        payload = json.load(sys.stdin)
    else:
        with args.input.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("inference input must be a JSON object")
    allowed = SWING_IDS if args.stage == "swing" else RELEASE_IDS
    available = payload.get("available_ids")
    if available is not None:
        if not isinstance(available, list):
            raise ValueError("available_ids must be an array")
        available = [action_id(value, "available_id", allowed) for value in available]
    if args.stage == "swing":
        swing_id = policy.choose_swing(SwingState.from_dict(payload.get("state")), available)
        lift, pitch = swing_levels(swing_id)
        return {"swing_id": swing_id, "lift_level": lift, "pitch_level": pitch}
    swing_id = action_id(payload.get("swing_id"), "swing_id", SWING_IDS)
    release_id = policy.choose_release(ReleaseState.from_dict(payload.get("state")), swing_id, available)
    return {"release_id": release_id, "timing_level": release_id - 1}


def _evaluate_sim(args):
    policies = load_policies(args.model)
    results = {name: Counter() for name in policies}
    rewards = {name: [] for name in policies}
    paired = Counter()
    rng = random.Random(args.seed)
    for index in range(args.trials):
        rollout_seed = rng.getrandbits(64)
        succeeded = {}
        for name, policy in policies.items():
            try:
                trial = simulate_trial(policy, rollout_seed, f"eval-{index + 1}", "eval")
                result = score(trial.outcome)
                label = trial.metadata.get("abort_reason", result.result)
                reward, success = result.reward, result.success
            except UnsupportedStateError:
                label, reward, success = "unsupported_swing_state", 0.0, False
            results[name][label] += 1
            rewards[name].append(reward)
            succeeded[name] = success
        if succeeded["iql"] and not succeeded["fixed"]:
            paired["only_iql_succeeded"] += 1
        elif succeeded["fixed"] and not succeeded["iql"]:
            paired["only_fixed_succeeded"] += 1

    return {
        "simulator": "planar_v1",
        "seed": args.seed,
        "trials_per_policy": args.trials,
        "policies": {
            name: {
                "successes": counts["success"],
                "success_rate": counts["success"] / args.trials,
                "mean_reward": mean(rewards[name]),
                "outcomes": dict(sorted(counts.items())),
            }
            for name, counts in results.items()
        },
        "paired": {key: paired[key] for key in ("only_iql_succeeded", "only_fixed_succeeded")},
    }


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)

    simulate = commands.add_parser("simulate", help="collect a balanced planar simulation dataset")
    simulate.add_argument("--output", required=True, type=Path)
    simulate.add_argument("--sessions", type=_positive, default=8)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.set_defaults(run=_simulate)

    train = commands.add_parser("train", help="fit IQL and a fixed baseline from JSONL trials")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--seed", type=int, default=7, help="recording session split seed")
    train.add_argument("--expectile", type=float, default=0.75)
    train.add_argument("--beta", type=float, default=5.0, help="inverse advantage temperature")
    train.add_argument("--all-data", action="store_true", help="fit all sessions without a holdout")
    train.set_defaults(run=_train)

    infer = commands.add_parser("infer", help="choose one action from an observation JSON object")
    infer.add_argument("--model", required=True, type=Path)
    infer.add_argument("--stage", choices=("swing", "release"), required=True)
    infer.add_argument("--policy", choices=("iql", "fixed"), default="iql")
    infer.add_argument("--input", type=Path, help="read this JSON file instead of standard input")
    infer.set_defaults(run=_infer)

    evaluate = commands.add_parser("evaluate-sim", help="compare trained policies on fresh paired rollouts")
    evaluate.add_argument("--model", required=True, type=Path)
    evaluate.add_argument("--trials", type=_positive, default=400)
    evaluate.add_argument("--seed", type=int, default=202)
    evaluate.set_defaults(run=_evaluate_sim)

    args = parser.parse_args()
    try:
        _dump(args.run(args), sys.stdout)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
