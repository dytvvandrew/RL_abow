import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from bottle_flip.umil import Episode, ROBOT_TYPE, read_episode


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "install requirements-umil.txt")
class UmilReaderTest(unittest.TestCase):
    def write_episode(self, root, version, *, invalid=False):
        import pyarrow as pa
        import pyarrow.parquet as pq

        (root / "meta").mkdir()
        info = {"robot_type": ROBOT_TYPE, "codebase_version": version, "fps": 30, "chunks_size": 1}
        (root / "meta" / "info.json").write_text(json.dumps(info))
        file = (root / "data/chunk-001/episode_000001.parquet" if version == "v2.1"
                else root / "data/chunk-000/file-000.parquet")
        file.parent.mkdir(parents=True)
        data = {
            "episode_index": [1, 1],
            "frame_index": [1, 0],
            "timestamp": [1 / 30, 0.0],
            "source.right.controller.timestamp_ns": [123_000_000, 90_000_000],
            "valid.right.controller": [not invalid, True],
            "observation.right.ee_pose": [[0.1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]],
        }
        if version == "v3.0":
            for key, values in data.items():
                values.append(0 if key in ("episode_index", "frame_index") else values[0])
        pq.write_table(pa.table(data), file)

    def test_reads_v21_and_filters_consolidated_v30(self):
        for version in ("v2.1", "v3.0"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_episode(root, version)
                episode = read_episode(root, 1, "right")
                self.assertEqual(episode.describe()["usable_frames"], 2)
                np.testing.assert_array_equal(episode.frame, [0, 1])
                np.testing.assert_allclose(episode.ee_pose[:, 0], [0.0, 0.1])
                self.assertEqual(episode.controller_ns.tolist(), [90_000_000, 123_000_000])

    def test_invalid_tracking_is_reported_and_rejected_for_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_episode(root, "v3.0", invalid=True)
            episode = read_episode(root, 1, "right")
            self.assertEqual(episode.describe()["usable_frames"], 1)
            with self.assertRaisesRegex(ValueError, "frame 1"):
                episode.window(0, 2)


@unittest.skipUnless(importlib.util.find_spec("scipy"), "install requirements-umil.txt")
class UmilRetargetTest(unittest.TestCase):
    def assets(self, root):
        folder = root / "urdf"
        folder.mkdir()
        model = ET.Element("robot", name="unit-test")
        parent = "base_link"
        for i, axis in enumerate(("0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"), 1):
            j = ET.SubElement(model, "joint", name=f"Joint{i}_R", type="revolute")
            child = f"Link{i}_R"
            ET.SubElement(j, "origin", xyz="0.08 0 0.03", rpy="0 0 0")
            ET.SubElement(j, "parent", link=parent)
            ET.SubElement(j, "child", link=child)
            ET.SubElement(j, "axis", xyz=axis)
            ET.SubElement(j, "limit", lower="-2", upper="2", velocity="1")
            parent = child
        tcp = ET.SubElement(model, "joint", name="JointTCP_R", type="fixed")
        ET.SubElement(tcp, "origin", xyz="0.06 0 0", rpy="0 0 0")
        ET.SubElement(tcp, "parent", link=parent)
        ET.SubElement(tcp, "child", link="TCP_Link_R")
        path = folder / "Stand-m6.urdf"
        ET.ElementTree(model).write(path)
        return path

    def setup_sample(self, root, *, jump=False, absolute=False):
        from scipy.spatial.transform import Rotation
        from bottle_flip.robot_kinematics import Tianji
        from bottle_flip.umil_retarget import Calibration

        rig = Tianji(self.assets(root), np.eye(4))
        seed = np.array([0.4, -0.5, 0.2, 0.8, -0.35, 0.3, 0.1])
        ee_from_grasp = np.eye(4)
        ee_from_grasp[:3, :3] = Rotation.from_euler("x", 0.4).as_matrix()
        ee_from_grasp[:3, 3] = [0.03, -0.01, 0.08]
        flange_from_grasp = np.eye(4)
        flange_from_grasp[:3, 3] = [0.09, 0.015, 0.02]
        base_from_recording = np.eye(4)
        base_from_recording[:3, :3] = Rotation.from_euler("z", 0.6).as_matrix()
        base_from_recording[:3, 3] = [0.23, 0.15, -0.11]
        path = np.tile(seed, (3, 1))
        path[:, 0] += [0.0, 0.012 if not jump else 1.1, 0.024]
        flange = np.asarray([rig.evaluate(q)[0] for q in path])
        ee = np.linalg.inv(base_from_recording) @ flange @ flange_from_grasp @ np.linalg.inv(ee_from_grasp)
        pose = np.column_stack((ee[:, :3, 3], Rotation.from_matrix(ee[:, :3, :3]).as_quat()))
        episode = Episode("v3.0", 30, 0, "right", np.arange(3), np.arange(3) / 30,
                          np.arange(3) + 1, pose, np.ones(3, dtype=bool))
        calibration = Calibration("right", "example", seed, ee_from_grasp, flange_from_grasp,
                                  base_from_recording if absolute else None)
        return episode, calibration, flange

    def test_relative_mapping_preserves_grasp_frame_and_joint_rates(self):
        from bottle_flip.umil_retarget import retarget_episode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, calibration, flange = self.setup_sample(root)
            plan = retarget_episode(episode, root, calibration)
            self.assertEqual(plan["status"], "kinematic_candidate")
            self.assertEqual(plan["flange_frame"], "TCP_Link_R")
            np.testing.assert_allclose(plan["target_flange_pose_base"], flange, atol=1e-10)
            np.testing.assert_allclose(plan["joints_rad"][0], calibration.seed_joints_rad)
            self.assertLess(np.max(plan["diagnostics"]["peak_joint_speed_rad_s"]), 1.000001)

    def test_unreachable_at_recorded_speed_does_not_export_joint_commands(self):
        from bottle_flip.umil_retarget import retarget_episode

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode, calibration, _ = self.setup_sample(root, jump=True)
            plan = retarget_episode(episode, root, calibration)
            self.assertEqual(plan["status"], "infeasible")
            self.assertEqual(plan["diagnostics"]["first_failed_frame"], 1)
            self.assertNotIn("joints_rad", plan)

    def test_absolute_mapping_uses_measured_recording_to_base_transform(self):
        from bottle_flip.umil_retarget import flange_targets

        with tempfile.TemporaryDirectory() as directory:
            episode, calibration, flange = self.setup_sample(Path(directory), absolute=True)
            targets = flange_targets(episode.ee_pose, calibration, np.eye(4))
            np.testing.assert_allclose(targets, flange, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
