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
    def __init__(self, urdf, base):
        self.urdf = ET.parse(urdf).getroot()
        self.base = base
        self.chain = [Joint.parse(element) for element in self.urdf.findall("joint")
                      if element.attrib["name"].endswith("_R")]
        self.lower, self.upper = np.asarray([joint.limits for joint in self.chain[:-1]]).T

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
GRIP_CENTER = np.array([0.055, 0.027, 0.098])
OPEN = np.array([0.24, 0.00, 0.00, 0.00, 0.00, 0.00])
ACTUATED = ("right_thumb_metacarpal_joint", *(f"right_{digit}_proximal_joint" for digit in DIGITS))


class Revo2Pads:
    def __init__(self):
        urdf = ET.parse(ASSET_DIRECTORY / "urdf" / "revo2_right_hand.urdf").getroot()
        self.joints = [Joint.parse(element) for element in urdf.findall("joint")]
        self.near = self._fit_aperture(0.0435)
        self.closed = self._fit_aperture(0.0395)

    def evaluate(self, curl):
        if curl <= 0.85:
            commands = OPEN + (curl / 0.85) * (self.near - OPEN)
        else:
            commands = self.near + ((curl - 0.85) / 0.15) * (self.closed - self.near)
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
