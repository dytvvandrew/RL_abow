"""URDF kinematics for the simulated right arm and Revo2 fingertips."""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.spatial.transform import Rotation

from .revo2_preview import ASSET_DIRECTORY


def vector(element, attribute):
    return np.fromstring(element.attrib[attribute], sep=" ")


def origin(element):
    pose = np.eye(4)
    pose[:3, 3] = vector(element, "xyz")
    pose[:3, :3] = Rotation.from_euler("xyz", vector(element, "rpy")).as_matrix()
    return pose


def rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    pose: np.ndarray
    axis: np.ndarray | None
    limits: tuple[float, float] | None
    mimic: tuple[str, float, float] | None

    @classmethod
    def parse(cls, element):
        limit, mimic = element.find("limit"), element.find("mimic")
        return cls(
            element.attrib["name"],
            element.find("parent").attrib["link"],
            element.find("child").attrib["link"],
            origin(element.find("origin")),
            vector(element.find("axis"), "xyz") if element.attrib["type"] == "revolute" else None,
            (float(limit.attrib["lower"]), float(limit.attrib["upper"])) if limit is not None else None,
            (mimic.attrib["joint"], float(mimic.attrib["multiplier"]), float(mimic.attrib["offset"]))
            if mimic is not None else None,
        )

    def relative_pose(self, angle):
        pose = self.pose.copy()
        if self.axis is not None:
            pose[:3, :3] = pose[:3, :3] @ rotation(self.axis, angle)
        return pose


class Tianji:
    def __init__(self, urdf, base, side="R"):
        if side not in ("L", "R"):
            raise ValueError("Stand-m6 arm side must be L or R")
        self.urdf = ET.parse(urdf).getroot()
        self.base = base
        self.side = side
        arm = [element for element in self.urdf.findall("joint")
               if element.attrib["name"].endswith(f"_{side}")]
        self.chain = [Joint.parse(element) for element in arm]
        self.lower, self.upper = np.asarray([joint.limits for joint in self.chain[:-1]]).T
        self.speed_limits = np.asarray([float(element.find("limit").attrib["velocity"])
                                        for element in arm if element.attrib["type"] == "revolute"])

    @property
    def forward_axis(self):
        # The factory Stand frame uses -Y forward and +X left.
        return -self.base[:3, 1]

    def apply_m6_sdk_limits(self, path):
        """Intersect URDF limits with the delivered TYPE1017 M6 configuration."""
        rows = [[float(value) for value in line.split(",") if value.strip()]
                for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 26 or rows[0][0] != 1017 or rows[13][0] != 1017:
            raise ValueError(f"{path}: expected two TYPE1017 M6 arm blocks")
        first = 14 if self.side == "R" else 1
        limits = np.deg2rad(np.asarray([row[4:7] for row in rows[first:first + 7]])).T
        if limits.shape != (3, 7) or not np.isfinite(limits).all() or np.any(limits[2] <= 0):
            raise ValueError(f"{path}: invalid arm {self.side} limits")
        self.lower = np.maximum(self.lower, limits[0])
        self.upper = np.minimum(self.upper, limits[1])
        self.speed_limits = np.minimum(self.speed_limits, limits[2])
        if np.any(self.lower >= self.upper):
            raise ValueError(f"{path}: SDK and URDF position limits do not overlap")

    def upper_arm_vectors(self, angles):
        """Return world-space shoulder-to-elbow vectors for joint samples."""
        shoulder, elevation, elbow = self.chain[:3]
        upper = elbow.pose[:3, 3]
        swing = np.stack([rotation(elevation.axis, angle) @ upper
                          for angle in angles[:, 1]])
        swing = swing @ elevation.pose[:3, :3].T
        rotated = np.stack([rotation(shoulder.axis, angle) @ vector
                            for angle, vector in zip(angles[:, 0], swing)])
        return rotated @ (self.base @ shoulder.pose)[:3, :3].T

    def evaluate(self, angles):
        pose = self.base
        links, centers, axes = {}, [], []
        for joint, angle in zip(self.chain, (*angles, 0.0)):
            pose = pose @ joint.pose
            if joint.axis is not None:
                centers.append(pose[:3, 3])
                axes.append(pose[:3, :3] @ joint.axis)
                pose[:3, :3] = pose[:3, :3] @ rotation(joint.axis, angle)
            links[joint.child] = pose
        jacobian = np.concatenate((np.cross(axes, pose[:3, 3] - centers), axes), axis=1).T
        return pose, jacobian, links

    @staticmethod
    def error(target, actual):
        delta = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
        return np.concatenate((target[:3, 3] - actual[:3, 3], delta))

    def initialize(self, target, previous=None):
        weights = np.array([2, 2, 2, 0.55, 0.55, 0.55])
        rng = np.random.default_rng(7)
        guesses = [np.zeros(7)] if previous is None else [previous]
        guesses.extend(rng.uniform(self.lower * 0.75, self.upper * 0.75, size=(12, 7)))
        for guess in guesses:
            fit = least_squares(
                lambda angles: weights * self.error(target, self.evaluate(angles)[0]),
                guess, bounds=(self.lower, self.upper), ftol=1e-8, gtol=1e-8, max_nfev=80,
            )
            if np.linalg.norm(fit.fun) < 2e-4:
                return fit.x
        raise ValueError("initial Revo2 wrist position is outside the Tianji IK workspace")

    def follow(self, target, angles):
        for _ in range(10):
            pose, jacobian, links = self.evaluate(angles)
            error = self.error(target, pose)
            if np.linalg.norm(error) < 2e-5:
                return angles, pose, links
            jacobian[:3] *= 2.5
            error[:3] *= 2.5
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(6) * 0.0003, error)
            angles = np.clip(angles + np.clip(delta, -0.20, 0.20), self.lower, self.upper)
        pose, _, links = self.evaluate(angles)
        if np.linalg.norm(self.error(target, pose)) > 0.003:
            angles = self.initialize(target, angles)
            pose, _, links = self.evaluate(angles)
        return angles, pose, links


DIGITS = ("thumb", "index", "middle", "ring", "pinky")
PAD_OFFSETS = np.asarray([
    [0.0, 0.002, -0.006], [-0.0037, 0.0013, -0.0003],
    [-0.0039, 0.0, 0.0016], [-0.0037, 0.0, 0.0017], [-0.0036, 0.0, 0.0004],
])
PAD_RADIUS = 0.009
CAP_RADIUS = 0.0155
CAP_CENTER_HEIGHT = 0.096
GRIP_CENTER = np.array([0.055, 0.027, 0.098])
OPEN = np.array([0.24, 0.00, 0.00, 0.00, 0.00, 0.00])
ACTUATED = ("right_thumb_metacarpal_joint", *(f"right_{digit}_proximal_joint" for digit in DIGITS))


def _revo2_joints():
    urdf = ET.parse(ASSET_DIRECTORY / "urdf" / "revo2_right_hand.urdf").getroot()
    return [Joint.parse(element) for element in urdf.findall("joint")]


class Revo2Pads:
    def __init__(self):
        self.joints = _revo2_joints()
        self.near = self._fit_aperture(0.0435)
        self.closed = self._fit_aperture(0.0395)

    def _commands(self, curl):
        if curl <= 0.85:
            return OPEN + (curl / 0.85) * (self.near - OPEN)
        return self.near + ((curl - 0.85) / 0.15) * (self.closed - self.near)

    def evaluate(self, curl, index_curl=None, thumb_press=0.0):
        commands = self._commands(curl)
        if index_curl is not None:
            commands[2] = self._commands(index_curl)[2]
        # The metacarpal sweeps the thumb down along the bottle. A small
        # flexion keeps the pad in contact as the metacarpal rotates.
        commands[0] += thumb_press
        commands[1] += 0.12 * thumb_press
        return self._forward(commands)

    def _fit_aperture(self, radius):
        commands = OPEN.copy()
        commands[0] = 1.10
        for index in range(5):
            def gap(angle):
                commands[index + 1] = angle
                _, pads = self._forward(commands)
                radial = (pads[index] - GRIP_CENTER)[[0, 2]]
                return np.linalg.norm(radial) - radius
            commands[index + 1] = brentq(gap, 0.0, 0.5)
        return commands

    def _forward(self, commands):
        angles = dict(zip(ACTUATED, commands))
        poses = {"right_base_link": np.eye(4)}
        for joint in self.joints:
            if joint.mimic is not None:
                parent, scale, offset = joint.mimic
                angles[joint.name] = np.clip(angles[parent] * scale + offset, *joint.limits)
            poses[joint.child] = poses[joint.parent] @ joint.relative_pose(angles.get(joint.name, 0.0))
        pads = np.asarray([
            poses[f"right_{digit}_touch_link"][:3, :3] @ offset + poses[f"right_{digit}_touch_link"][:3, 3]
            for digit, offset in zip(DIGITS, PAD_OFFSETS)
        ])
        return angles, pads


class Revo2CapPinch(Revo2Pads):
    """Oppose two pads across the cap and curl the other fingers away from the bottle."""

    def __init__(self):
        self.joints = _revo2_joints()
        parked = np.full(3, 1.30)
        angle = math.radians(35.0)
        direction = np.array([math.sin(angle), 0.0, -math.cos(angle)])
        target = 2.0 * (CAP_RADIUS + PAD_RADIUS - 0.0009) * direction
        actuated_limits = {joint.name: joint.limits for joint in self.joints if joint.name in ACTUATED}

        def error(commands):
            pads = self._forward(np.r_[commands, parked])[1]
            return pads[0] - pads[1] - target

        limits = np.asarray([actuated_limits[name] for name in ACTUATED[:3]]).T
        fit = least_squares(error, [1.25, 0.30, 0.44], bounds=limits, gtol=1e-11)
        if np.linalg.norm(fit.fun) > 0.0001:
            raise ValueError("Revo2 thumb and index cannot oppose across the cap")
        self.closed = np.r_[fit.x, parked]
        self.open = self.closed - np.array([0.0, 0.16, 0.15, 0.0, 0.0, 0.0])
        self.center = self._forward(self.closed)[1][:2].mean(axis=0)

    def evaluate(self, curl, index_curl=None, thumb_press=0.0):
        return self._forward(self.open + curl * (self.closed - self.open))
