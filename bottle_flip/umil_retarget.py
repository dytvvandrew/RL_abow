"""Map calibrated UMI-Lite grasp poses to a constrained Stand-m6 joint path."""

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from .limited_throw import WRIST_COUPLING_LIMIT, WRIST_COUPLING_SCALE
from .robot_kinematics import Tianji


POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
WRIST_ROWS = np.zeros((4, 7))
WRIST_ROWS[:, 5] = WRIST_COUPLING_SCALE * np.array([1.0, 1.0, -1.0, -1.0])
WRIST_ROWS[:, 6] = [1.0, -1.0, 1.0, -1.0]


def _transform(value, field):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{field}: expected a finite 4 x 4 transform")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError(f"{field}: expected a proper SE(3) transform")
    return matrix


@dataclass(frozen=True)
class Calibration:
    side: str
    source: str
    seed_joints_rad: np.ndarray
    ee_from_grasp: np.ndarray
    flange_from_grasp: np.ndarray
    base_from_recording: np.ndarray | None = None

    @classmethod
    def load(cls, path, side):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["side"] != side:
            raise ValueError("calibration side must match the recorded hand")
        source = data["calibration_source"]
        if source not in ("example", "measured"):
            raise ValueError("calibration_source must be example or measured")
        seed = np.asarray(data["seed_joints_rad"], dtype=float)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("seed_joints_rad must contain seven finite joint angles in radians")
        base = data.get("base_from_recording")
        return cls(side, source, seed, _transform(data["ee_from_grasp"], "ee_from_grasp"),
                   _transform(data["flange_from_grasp"], "flange_from_grasp"),
                   None if base is None else _transform(base, "base_from_recording"))


def flange_targets(ee_pose, calibration, base_from_seed_flange):
    recording_from_ee = np.broadcast_to(np.eye(4), (len(ee_pose), 4, 4)).copy()
    recording_from_ee[:, :3, :3] = Rotation.from_quat(ee_pose[:, 3:]).as_matrix()
    recording_from_ee[:, :3, 3] = ee_pose[:, :3]
    recording_from_grasp = recording_from_ee @ calibration.ee_from_grasp
    base_from_recording = calibration.base_from_recording
    if base_from_recording is None:
        base_from_seed_grasp = base_from_seed_flange @ calibration.flange_from_grasp
        base_from_recording = base_from_seed_grasp @ np.linalg.inv(recording_from_grasp[0])
    grasp_from_flange = np.linalg.inv(calibration.flange_from_grasp)
    return base_from_recording @ recording_from_grasp @ grasp_from_flange


def _in_limits(q, lower, upper):
    epsilon = 1e-8
    return bool(np.isfinite(q).all() and np.all(q >= lower - epsilon)
                and np.all(q <= upper + epsilon)
                and np.all(WRIST_ROWS @ q <= WRIST_COUPLING_LIMIT + epsilon))


def _pose_error(rig, target, q):
    error = rig.error(target, rig.evaluate(q)[0])
    return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))


def _solve_frame(rig, target, seed, lower, upper):
    def cost(q):
        position, orientation = _pose_error(rig, target, q)
        delta = q - seed
        return ((position / POSITION_TOLERANCE_M) ** 2
                + (orientation / ORIENTATION_TOLERANCE_RAD) ** 2 + 0.01 * (delta @ delta))

    if cost(seed) < 1e-12:
        return seed
    result = minimize(cost, seed, method="SLSQP", bounds=list(zip(lower, upper)),
                      constraints=[LinearConstraint(WRIST_ROWS, -np.inf, WRIST_COUPLING_LIMIT)],
                      options={"maxiter": 60, "ftol": 1e-10})
    return result.x


def retarget_episode(episode, arm_assets, calibration, *, sdk_limits=None, time_scale=1.0):
    if calibration.side != episode.side:
        raise ValueError("calibration side must match the episode side")
    if len(episode.frame) < 2 or not episode.usable.all():
        raise ValueError("select a tracked window of at least two frames")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time-scale must be finite and positive")
    dt_source = np.diff(episode.timestamp)
    if np.any(dt_source <= 0) or np.any(dt_source > 1.5 / episode.fps):
        raise ValueError("selected window contains a discontinuity in playback time")

    side = "R" if episode.side == "right" else "L"
    rig = Tianji(arm_assets / "urdf" / "Stand-m6.urdf", np.eye(4), side=side)
    if sdk_limits is None:
        delivered = arm_assets.parent / "marvin_sdk" / "CommonConfig" / "ccs_m6_40.MvKDCfg"
        sdk_limits = delivered if delivered.is_file() else None
    if sdk_limits is not None:
        rig.apply_m6_sdk_limits(sdk_limits)
    seed = calibration.seed_joints_rad
    if not _in_limits(seed, rig.lower, rig.upper):
        raise ValueError("seed_joints_rad exceeds the Stand-m6 position or wrist-coupling limits")
    targets = flange_targets(episode.ee_pose, calibration, rig.evaluate(seed)[0])
    times = (episode.timestamp - episode.timestamp[0]) * time_scale
    dt = np.diff(times)
    if not np.isfinite(times).all() or np.any(dt <= 0):
        raise ValueError("time-scale yields invalid output timestamps")
    report = {
        "schema_version": 1,
        "source": episode.describe(),
        "calibration_source": calibration.source,
        "mapping": "relative_to_seed" if calibration.base_from_recording is None else "absolute",
        "output_frame": "Stand-m6 base_link",
        "flange_frame": f"TCP_Link_{side}",
        "joint_names": [joint.name for joint in rig.chain[:-1]],
        "time_scale": time_scale,
        "limits": {
            "delivered_sdk_applied": sdk_limits is not None,
            "joint_speed_rad_s": rig.speed_limits.tolist(),
            "wrist_coupling_limit_deg": math.degrees(WRIST_COUPLING_LIMIT),
        },
    }
    frames = []
    errors = []
    for i, target in enumerate(targets):
        lower, upper = rig.lower, rig.upper
        if i:
            step_limit = dt[i - 1] * rig.speed_limits
            lower = np.maximum(lower, seed - step_limit)
            upper = np.minimum(upper, seed + step_limit)
        q = _solve_frame(rig, target, seed, lower, upper)
        within = _in_limits(q, lower, upper)
        position, orientation = _pose_error(rig, target, q) if within else (None, None)
        if not within or position > POSITION_TOLERANCE_M or orientation > ORIENTATION_TOLERANCE_RAD:
            return {
                **report, "status": "infeasible", "mapped_frames": i,
                "diagnostics": {
                    "first_failed_frame": int(episode.frame[i]),
                    "reason": "cannot track target within pose and joint limits",
                    "position_error_m": position,
                    "orientation_error_deg": None if orientation is None else math.degrees(orientation),
                },
            }
        frames.append(q)
        errors.append([position, orientation])
        seed = q

    q = np.asarray(frames)
    velocity = np.diff(q, axis=0) / dt[:, None]
    accel = np.diff(velocity, axis=0) / ((dt[:-1] + dt[1:]) / 2)[:, None]
    max_position_error, max_orientation_error = np.max(errors, axis=0)
    return {
        **report, "status": "kinematic_candidate", "mapped_frames": len(q),
        "diagnostics": {
            "max_position_error_m": float(max_position_error),
            "max_orientation_error_deg": math.degrees(max_orientation_error),
            "peak_joint_speed_rad_s": np.max(np.abs(velocity), axis=0).tolist(),
            "peak_joint_accel_rad_s2": np.max(np.abs(accel), axis=0).tolist() if len(accel) else None,
        },
        "source_frame_indices": episode.frame.tolist(),
        "source_playback_timestamp_s": episode.timestamp.tolist(),
        "source_controller_timestamp_ns": episode.controller_ns.tolist(),
        "time_s": times.tolist(),
        "target_flange_pose_base": targets.tolist(),
        "joints_rad": q.tolist(),
    }
