"""Animate the official Revo2 visual geometry in an Isaac Sim stage."""

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "assets" / "revo2"
GRIP_CENTER = (0.039, -0.007, 0.083)
GRIP_YAW = 0.91
OPEN_ANGLES = {"right_thumb_metacarpal_joint": 0.16}
CLOSED_ANGLES = {
    "right_thumb_metacarpal_joint": 0.96,
    "right_thumb_proximal_joint": 0.50,
    "right_index_proximal_joint": 0.42,
    "right_middle_proximal_joint": 0.45,
    "right_ring_proximal_joint": 0.46,
    "right_pinky_proximal_joint": 0.42,
}


def _numbers(element, attribute):
    return tuple(float(number) for number in element.attrib[attribute].split())


def _rotation(axis, angle):
    from pxr import Gf

    return Gf.Quatd(math.cos(angle / 2), Gf.Vec3d(*axis) * math.sin(angle / 2))


def _origin(xform, origin):
    from pxr import Gf, UsdGeom

    roll, pitch, yaw = _numbers(origin, "rpy")
    rotation = (
        _rotation((0, 0, 1), yaw) * _rotation((0, 1, 0), pitch) * _rotation((1, 0, 0), roll)
    )
    xform.AddTranslateOp().Set(Gf.Vec3d(*_numbers(origin, "xyz")))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation)


@lru_cache(maxsize=None)
def _read_mesh(path):
    payload = path.read_bytes()
    count = int.from_bytes(payload[80:84], "little")
    if len(payload) != 84 + 50 * count:
        raise ValueError(f"invalid binary STL: {path}")
    facet = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("tag", "<u2")])
    faces = np.frombuffer(payload, dtype=facet, offset=84)
    points, indices = np.unique(faces["vertices"].reshape(-1, 3), axis=0, return_inverse=True)
    return points, indices.reshape(-1).astype(np.int32), np.ascontiguousarray(faces["normal"])


def add_stl(stage, path, asset, color):
    from pxr import Gf, UsdGeom, Vt

    points, indices, normals = _read_mesh(asset)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(normals), 3, dtype=np.int32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return mesh


class Revo2Preview:
    def __init__(self, stage, path):
        from pxr import UsdGeom

        model = ET.parse(ASSET_DIRECTORY / "urdf" / "revo2_right_hand.urdf").getroot()
        self.root = UsdGeom.Xform.Define(stage, path)
        self.position = self.root.AddTranslateOp()
        self.orientation = self.root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        self.palm_rotation = _rotation((0, 0, 1), GRIP_YAW) * _rotation((1, 0, 0), math.pi / 2)
        self.last_curl = None
        self.joints = []
        link_paths = {"right_base_link": f"{path}/right_base_link"}
        UsdGeom.Xform.Define(stage, link_paths["right_base_link"])
        for joint in model.findall("joint"):
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            child_path = f"{link_paths[parent]}/{child}"
            origin = UsdGeom.Xform.Define(stage, child_path)
            _origin(origin, joint.find("origin"))
            if joint.attrib["type"] == "revolute":
                child_path += "/Rotation"
                transform = UsdGeom.Xform.Define(stage, child_path)
                orient = transform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
                axis = _numbers(joint.find("axis"), "xyz")
                self.joints.append((joint, orient, axis))
            link_paths[child] = child_path
        for link in model.findall("link"):
            name = link.attrib["name"]
            for index, visual in enumerate(link.findall("visual")):
                color = (0.11, 0.16, 0.19) if "touch" in name else (0.71, 0.77, 0.82)
                filename = visual.find("geometry/mesh").attrib["filename"]
                asset = ASSET_DIRECTORY / filename.removeprefix("package://revo2_description/")
                mesh = add_stl(stage, f"{link_paths[name]}/Visual_{index}", asset, color)
                _origin(mesh, visual.find("origin"))

    def set_joint_angles(self, angles):
        for joint, orient, axis in self.joints:
            orient.Set(_rotation(axis, angles[joint.attrib["name"]]))

    def pose(self, center, pitch, curl, offset=(0, 0, 0)):
        from pxr import Gf

        orientation = _rotation((0, 1, 0), pitch) * self.palm_rotation
        grip = Gf.Rotation(orientation).TransformDir(Gf.Vec3d(*GRIP_CENTER))
        self.position.Set(Gf.Vec3d(*center) + Gf.Vec3d(*offset) - grip)
        self.orientation.Set(orientation)
        if curl == self.last_curl:
            return
        self.last_curl = curl
        angles = {}
        for joint, orient, axis in self.joints:
            name = joint.attrib["name"]
            mimic = joint.find("mimic")
            if mimic is None:
                open_angle = OPEN_ANGLES.get(name, 0.0)
                angle = open_angle + curl * (CLOSED_ANGLES[name] - open_angle)
            else:
                angle = (
                    angles[mimic.attrib["joint"]] * float(mimic.attrib["multiplier"])
                    + float(mimic.attrib["offset"])
                )
            limit = joint.find("limit").attrib
            angle = min(float(limit["upper"]), max(float(limit["lower"]), angle))
            angles[name] = angle
            orient.Set(_rotation(axis, angle))
