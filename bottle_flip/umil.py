"""Read UMI-Lite LeRobot episodes and map their poses to an offline arm plan."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


ROBOT_TYPE = "umi_lite_bimanual_six_joint_hands"


@dataclass(frozen=True)
class Episode:
    version: str
    fps: float
    index: int
    side: str
    frame: np.ndarray
    timestamp: np.ndarray
    controller_ns: np.ndarray
    ee_pose: np.ndarray
    usable: np.ndarray

    def describe(self):
        return {
            "robot_type": ROBOT_TYPE,
            "lerobot_version": self.version,
            "episode_index": self.index,
            "side": self.side,
            "pose_field": f"observation.{self.side}.ee_pose",
            "pose_format": "OpenXR LOCAL (x_m, y_m, z_m, qx, qy, qz, qw)",
            "fps": self.fps,
            "frames": len(self.frame),
            "usable_frames": int(self.usable.sum()),
            "duration_s": float(self.timestamp[-1] - self.timestamp[0]),
        }

    def window(self, start, stop):
        stop = len(self.frame) if stop is None else stop
        if not 0 <= start < stop <= len(self.frame) or stop - start < 2:
            raise ValueError("select at least two frames with 0 <= start-frame < stop-frame <= episode length")
        selection = slice(start, stop)
        invalid = self.frame[selection][~self.usable[selection]]
        if len(invalid):
            raise ValueError(f"frame {invalid[0]} has no usable UMI-Lite tracking")
        return Episode(self.version, self.fps, self.index, self.side, self.frame[selection],
                       self.timestamp[selection], self.controller_ns[selection],
                       self.ee_pose[selection], self.usable[selection])


def _scalar(table, name, dtype):
    column = np.asarray(table[name].to_pylist(), dtype=dtype)
    if column.shape == (len(table), 1):
        column = column[:, 0]
    if column.shape != (len(table),) or table[name].null_count:
        raise ValueError(f"{name}: expected a scalar for every frame")
    return column


def read_episode(root, index, side):
    import pyarrow.dataset as arrow_dataset
    import pyarrow.parquet as parquet

    if side not in ("left", "right") or index < 0:
        raise ValueError("select a nonnegative episode index and the left or right hand")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if info.get("robot_type") != ROBOT_TYPE:
        raise ValueError(f"expected UMI-Lite robot_type {ROBOT_TYPE!r}")
    version = info["codebase_version"]
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid dataset fps")
    pose = f"observation.{side}.ee_pose"
    valid = f"valid.{side}.controller"
    source = f"source.{side}.controller.timestamp_ns"
    columns = ["episode_index", "frame_index", "timestamp", pose, valid, source]
    if version == "v2.1":
        chunk_size = int(info["chunks_size"])
        if chunk_size <= 0:
            raise ValueError("invalid v2.1 chunks_size")
        path = root / "data" / f"chunk-{index // chunk_size:03d}" / f"episode_{index:06d}.parquet"
        table = parquet.ParquetFile(path).read(columns=columns)
    elif version == "v3.0":
        dataset = arrow_dataset.dataset(root / "data", format="parquet")
        table = dataset.to_table(columns=columns, filter=arrow_dataset.field("episode_index") == index)
    else:
        raise ValueError(f"unsupported UMI-Lite LeRobot version {version!r}")
    if not len(table) or np.any(_scalar(table, "episode_index", np.int64) != index):
        raise ValueError(f"episode {index} is missing or contains mismatched episode indices")
    order = np.argsort(_scalar(table, "frame_index", np.int64))
    frames = _scalar(table, "frame_index", np.int64)[order]
    seconds = _scalar(table, "timestamp", float)[order]
    source_ns = _scalar(table, source, np.int64)[order]
    valid_rows = _scalar(table, valid, bool)[order]
    if not np.array_equal(frames, np.arange(len(table))):
        raise ValueError("episode frame indices must be continuous from zero")
    if not np.isfinite(seconds).all() or np.any(np.diff(seconds) <= 0):
        raise ValueError("episode playback timestamps must be finite and strictly increasing")
    poses = np.asarray([row if row is not None else [np.nan] * 7 for row in table[pose].to_pylist()],
                       dtype=float)[order]
    if poses.shape != (len(table), 7):
        raise ValueError(f"{pose}: expected three positions and a quaternion xyzw")
    norm = np.linalg.norm(poses[:, 3:], axis=1)
    usable = (valid_rows & (source_ns > 0) & np.isfinite(poses).all(axis=1)
              & (np.abs(norm - 1.0) <= 1e-3))
    return Episode(version, fps, index, side, frames, seconds, source_ns, poses, usable)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.",
                            dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="summarize a recorded episode")
    retarget = commands.add_parser("retarget", help="export a checked Stand-m6 joint trajectory")
    for command in (inspect, retarget):
        command.add_argument("--dataset", type=Path, required=True, help="task directory containing meta/ and data/")
        command.add_argument("--episode", type=int, required=True)
        command.add_argument("--side", choices=("left", "right"), default="right")
    retarget.add_argument("--arm-assets", type=Path, required=True, help="vendor Stand-m6 directory")
    retarget.add_argument("--sdk-limits", type=Path, help="override the delivered M6 configuration path")
    retarget.add_argument("--calibration", type=Path, required=True)
    retarget.add_argument("--output", type=Path, required=True)
    retarget.add_argument("--start-frame", type=int, default=0, help="first included frame")
    retarget.add_argument("--stop-frame", type=int, help="first excluded frame")
    retarget.add_argument("--time-scale", type=float, default=1.0, help="multiply recorded sample intervals")

    args = parser.parse_args()
    try:
        episode = read_episode(args.dataset, args.episode, args.side)
        if args.command == "inspect":
            print(json.dumps(episode.describe(), indent=2))
            return
        from .umil_retarget import Calibration, retarget_episode

        calibration = Calibration.load(args.calibration, args.side)
        episode = episode.window(args.start_frame, args.stop_frame)
        result = retarget_episode(episode, args.arm_assets, calibration,
                                  sdk_limits=args.sdk_limits, time_scale=args.time_scale)
        _write_json(args.output, result)
    except ImportError as error:
        parser.error(f"{error}; install requirements-umil.txt")
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("status", "mapped_frames", "diagnostics")}, indent=2))
    if result["status"] != "kinematic_candidate":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
