"""Tianji right arm and Revo2 contact-driven bottle flip in Isaac Sim 6.2."""

import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data import Outcome
from .limited_throw import DEFAULT_BOTTLE_COM_Z_M, DEFAULT_BOTTLE_MASS_KG, LimitedThrow
from .revo2_preview import Revo2Preview, add_stl
from .reward import score
from .robot_kinematics import CAP_CENTER_HEIGHT, CAP_RADIUS, GRIP_CENTER, PAD_RADIUS
from .robot_kinematics import Revo2CapPinch, Revo2Pads, Tianji, origin
from .simulator import BOTTLE_HALF_LENGTH_M as HALF, BOTTLE_RADIUS_M as RADIUS, angular_rate


DT = 1 / 240
LIFT_AT = 0.90
START = np.array([-0.10, 0.0, HALF])
PALM_MAP = Rotation.from_euler("z", 0.91).as_matrix() @ Rotation.from_euler("x", math.pi / 2).as_matrix()
TARGET_X = (0.105, 0.62)
TABLE_CENTER_X = 0.14
TABLE_HALF_SIZE = (0.525, 0.315, 0.033)
CAP_TABLE_HALF_LENGTH = 1.20
PAD_PATHS = [f"/World/Pad_{index}" for index in range(5)]
DEMO_PARAMETERS = Path(__file__).resolve().parents[1] / "assets" / "isaac_contact_flip.json"


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def table_layout(cap_pinch):
    half_x, half_y, _ = TABLE_HALF_SIZE
    if cap_pinch:
        return TABLE_CENTER_X - half_x + CAP_TABLE_HALF_LENGTH, (CAP_TABLE_HALF_LENGTH, half_y, TABLE_HALF_SIZE[2])
    return TABLE_CENTER_X, TABLE_HALF_SIZE


def on_landing_table(position, cap_pinch=False):
    center, (half_x, half_y, _) = table_layout(cap_pinch)
    return bool(abs(position[0] - center) <= half_x - RADIUS
                and abs(position[1]) <= half_y - RADIUS)


@dataclass(frozen=True)
class Throw:
    height: float = 0.53
    vz: float = 1.35
    vx: float = 0.48
    pitch: float = 0.06
    rate: float = 17.0
    duration: float = 0.60
    opening: float = 0.100
    inertia_growth: float = 1.15

    @property
    def open_at(self):
        return LIFT_AT + self.duration

    def command(self, time):
        end = np.array([0.035, 0.0, self.height, self.pitch])
        speed = np.array([self.vx, 0.0, self.vz, self.rate])
        start = np.append(START, 0.0)
        if time < LIFT_AT:
            approach = 1 - smooth(time / 0.32)
            pose = start + approach * np.array([0.0, 0.0, 0.17, 0.0])
            curl = smooth((time - 0.34) / 0.38)
        elif time < self.open_at:
            progress = (time - LIFT_AT) / self.duration
            pose = start + (end - start) * smooth(progress)
            pose += self.duration * speed * progress**2 * (progress - 1)
            curl = 1.0
        else:
            after = time - self.open_at
            follow = min(after, 0.082)
            pose = end + speed * follow
            pose[2] -= 4.905 * follow**2
            pose[3] = self.pitch + follow * angular_rate(follow / 1.5, self.rate, self.inertia_growth)
            # Withdraw along the bottle's axis while the opened fingers still
            # surround it. A lateral retreat here would sweep a pad through it.
            axial = np.array([math.sin(pose[3]), 0.0, math.cos(pose[3])])
            pose[:3] -= 0.235 * smooth((after - 0.021) / 0.061) * axial
            home = smooth((after - 0.082) / 0.45)
            pose += home * (np.array([-0.14, 0.21, 0.65, 0.35]) - pose)
            curl = 1 - smooth(after / self.opening)
        rx, ry, rz, pitch = pose
        wrist = np.eye(4)
        wrist[:3, :3] = Rotation.from_euler("y", pitch).as_matrix() @ PALM_MAP
        wrist[:3, 3] = np.array([rx, ry, rz]) - wrist[:3, :3] @ GRIP_CENTER
        return wrist, curl


def _stage(arm, bottle_start, show_target, forward_flick=False,
           bottle_mass=DEFAULT_BOTTLE_MASS_KG, bottle_com_z=DEFAULT_BOTTLE_COM_Z_M, cap_pinch=False):
    import omni.usd
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

    context = omni.usd.get_context()
    if context.get_stage() is None:
        context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physx = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx.CreateEnableCCDAttr(True)
    physx.CreateEnableEnhancedDeterminismAttr(True)

    def material(name, friction, stiffness=0):
        mat = UsdShade.Material.Define(stage, f"/World/{name}")
        surface = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        surface.CreateStaticFrictionAttr(friction)
        surface.CreateDynamicFrictionAttr(friction)
        surface.CreateRestitutionAttr(0.01)
        sdk = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        sdk.CreateFrictionCombineModeAttr("max")
        if stiffness:
            sdk.CreateCompliantContactStiffnessAttr(stiffness)
            sdk.CreateCompliantContactDampingAttr(85.0)
        return mat

    table_mat = material("TableContact", 0.85, stiffness=15_000)
    bottle_mat = material("PET", 0.55)
    fingertip_mat = material("TipRubber", 2.0, stiffness=25_000)

    def collide(geometry, surface):
        prim = geometry.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(surface, materialPurpose="physics")
        shape = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        shape.CreateContactOffsetAttr(0.001)
        shape.CreateRestOffsetAttr(0.0)

    def color(geometry, rgb):
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

    def cube(name, center, size, rgb, surface=None):
        geom = UsdGeom.Cube.Define(stage, f"/World/{name}")
        geom.CreateSizeAttr(1)
        geom.AddTranslateOp().Set(Gf.Vec3d(*center))
        geom.AddScaleOp().Set(Gf.Vec3f(*size))
        color(geom, rgb)
        if surface is not None:
            collide(geom, surface)

    cube("Floor", (0.0, 0, -0.889), (4, 4, 0.05), (0.14, 0.19, 0.23), table_mat)
    cube("Backdrop", (0.0, 1.65, 0.55), (4.0, 0.03, 2.9), (0.24, 0.32, 0.37))
    table_center, table_half_size = table_layout(cap_pinch)
    cube("Table", (table_center, 0, -table_half_size[2]),
         tuple(2 * half for half in table_half_size), (0.38, 0.46, 0.51), table_mat)
    cube("TableBase", (table_center if cap_pinch else 0.15, 0, -0.46),
         (1.80 if cap_pinch else 0.52, 0.40, 0.80), (0.20, 0.25, 0.29))
    pick_height = bottle_start[2] - HALF
    if pick_height > 0.002:
        cube("PickupRiser", (*bottle_start[:2], pick_height / 2),
             (0.14, 0.14, pick_height), (0.29, 0.43, 0.56), table_mat)
    if show_target:
        cube("Target", (sum(TARGET_X) / 2, 0, 0.0003),
             (TARGET_X[1] - TARGET_X[0], 0.32, 0.0006), (0.07, 0.48, 0.34))

    bottle = UsdGeom.Xform.Define(stage, "/World/Bottle")
    bottle.AddTranslateOp().Set(Gf.Vec3d(*bottle_start.tolist()))
    rigid = UsdPhysics.RigidBodyAPI.Apply(bottle.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(bottle.GetPrim())
    mass.CreateMassAttr(bottle_mass)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, bottle_com_z))
    sdk = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle.GetPrim())
    sdk.CreateEnableCCDAttr(True)
    sdk.CreateSolverPositionIterationCountAttr(24)
    sdk.CreateSolverVelocityIterationCountAttr(8)
    sdk.CreateAngularDampingAttr(0.0)
    sdk.CreateLinearDampingAttr(0.0)
    sdk.CreateMaxDepenetrationVelocityAttr(0.4)
    PhysxSchema.PhysxContactReportAPI.Apply(bottle.GetPrim()).CreateThresholdAttr(0.0)

    def cylinder(name, radius, height, z, rgb):
        geom = UsdGeom.Cylinder.Define(stage, f"/World/Bottle/{name}")
        geom.CreateAxisAttr(UsdGeom.Tokens.z)
        geom.CreateRadiusAttr(radius)
        geom.CreateHeightAttr(height)
        geom.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        color(geom, rgb)
        return geom

    if cap_pinch:
        bottle_color = (0.72, 0.82, 0.88)
        body = cylinder("Body", RADIUS, 0.165, -0.0225, bottle_color)
        collide(body, bottle_mat)
        shoulder = UsdGeom.Mesh.Define(stage, "/World/Bottle/Shoulder")
        sides = 24
        circle = [math.tau * i / sides for i in range(sides)]
        shoulder.CreatePointsAttr([
            Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), height)
            for radius, height in ((RADIUS, 0.060), (0.0125, 0.082)) for angle in circle
        ])
        faces = [[i, (i + 1) % sides, sides + (i + 1) % sides, sides + i] for i in range(sides)]
        faces.extend([list(reversed(range(sides))), list(range(sides, 2 * sides))])
        shoulder.CreateFaceVertexCountsAttr([len(face) for face in faces])
        shoulder.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
        shoulder.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        color(shoulder, bottle_color)
        collide(shoulder, bottle_mat)
        UsdPhysics.MeshCollisionAPI.Apply(shoulder.GetPrim()).CreateApproximationAttr("convexHull")
        collide(cylinder("Neck", 0.0125, 0.012, 0.086, bottle_color), bottle_mat)
        collide(cylinder("Cap", CAP_RADIUS, 0.018, CAP_CENTER_HEIGHT, (0.09, 0.53, 0.43)), bottle_mat)
        cylinder("WaterLevel", RADIUS + 0.00015, 0.070, -0.070, (0.10, 0.73, 0.63))
    else:
        hull = cylinder("Collider", RADIUS, HALF * 2, 0, (0.10, 0.67, 0.48))
        collide(hull, bottle_mat)
        hull.MakeInvisible()
        cylinder("WaterLevel", RADIUS, 0.070, -0.070, (0.10, 0.73, 0.63))
        cylinder("Body", RADIUS, 0.095, 0.0125, (0.72, 0.82, 0.88))
        cylinder("Shoulder", 0.025, 0.026, 0.073, (0.72, 0.82, 0.88))
        cylinder("Cap", 0.018, 0.019, 0.0955, (0.09, 0.53, 0.43))

    for index, path in enumerate(PAD_PATHS):
        pad = UsdGeom.Sphere.Define(stage, path)
        pad.CreateRadiusAttr(PAD_RADIUS)
        pad.AddTranslateOp().Set(Gf.Vec3d(0, 0, -1.5 - 0.03 * index))
        kinematic = UsdPhysics.RigidBodyAPI.Apply(pad.GetPrim())
        kinematic.CreateKinematicEnabledAttr(True)
        collide(pad, fingertip_mat)
        pad.MakeInvisible()

    handles = {}
    hand = None
    if arm is not None:
        rig, assets = arm

        def link_handle(name, rgb):
            link = UsdGeom.Xform.Define(stage, f"/World/Tianji/{name}")
            transform = link.AddTransformOp()
            element = rig.urdf.find(f"link[@name='{name}']")
            for index, visual in enumerate(element.findall("visual")):
                filename = Path(visual.find("geometry/mesh").attrib["filename"]).name
                geom = add_stl(stage, f"{link.GetPath()}/Visual_{index}", assets / "meshes" / filename, rgb)
                pose = origin(visual.find("origin"))
                geom.AddTransformOp().Set(Gf.Matrix4d(*pose.T.reshape(-1).tolist()))
            return transform

        for name in ["base_link"] + [joint.child for joint in rig.chain[:-1]]:
            handles[name] = link_handle(name, (0.56, 0.64, 0.70))
        handles["base_link"].Set(Gf.Matrix4d(*rig.base.T.reshape(-1).tolist()))
        if forward_flick:
            idle_left = Tianji(assets / "urdf" / "Stand-m6.urdf", rig.base, side="L")
            _, _, left_links = idle_left.evaluate(np.deg2rad([-90, 90, 0, -10, 0, 0, 0]))
            for name, pose in left_links.items():
                link_handle(name, (0.39, 0.44, 0.48)).Set(Gf.Matrix4d(*pose.T.reshape(-1).tolist()))
        base_rise = rig.base[2, 3] + 0.78
        if base_rise > 0:
            cube("ArmRiser", (*rig.base[:2, 3], -0.863 + base_rise / 2),
                 (0.69, 0.69, base_rise), (0.22, 0.27, 0.31))
        mount = UsdGeom.Xform.Define(stage, "/World/Tianji/Link7_R/TCP_Link_R")
        mount.AddTransformOp().Set(Gf.Matrix4d(*rig.chain[-1].pose.T.reshape(-1).tolist()))
        hand = Revo2Preview(stage, f"{mount.GetPath()}/Revo2")

    UsdLux.DomeLight.Define(stage, "/World/Fill").CreateIntensityAttr(480.0)
    light = UsdLux.DistantLight.Define(stage, "/World/Key")
    light.CreateIntensityAttr(1050.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(38, -23, -15))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(22 if forward_flick else 24)
    eye, target = Gf.Vec3d(1.32, -1.70, 0.93), Gf.Vec3d(0.11, 0.19, 0.22)
    if pick_height > 0.002:
        eye, target = Gf.Vec3d(1.65, -2.15, 1.43), Gf.Vec3d(0.08, 0.21, 0.43)
    if forward_flick:
        eye, target = Gf.Vec3d(0.35, -2.85, 1.28), Gf.Vec3d(-0.17, -0.09, 0.62)
    if cap_pinch:
        eye, target = Gf.Vec3d(1.85, -2.85, 1.36), Gf.Vec3d(0.35, -0.08, 0.63)
    up = Gf.Vec3d(0, 0, 1)
    camera_transform = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
    camera.AddTransformOp().Set(camera_transform)
    front = UsdLux.DistantLight.Define(stage, "/World/CameraFill")
    front.CreateIntensityAttr(900.0)
    front.AddTransformOp().Set(camera_transform)
    UsdLux.ShadowAPI.Apply(front.GetPrim()).CreateShadowEnableAttr(False)
    side = UsdLux.DistantLight.Define(stage, "/World/SideFill")
    side.CreateIntensityAttr(450.0)
    side.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(-1.4, -0.2, 1), target, up).GetInverse())
    UsdLux.ShadowAPI.Apply(side.GetPrim()).CreateShadowEnableAttr(False)
    return handles, hand, camera.GetPath()


class ContactTrial:
    def __init__(self, bottle, pads, contacts, hand_model):
        self.bottle, self.pads, self.contacts, self.hand_model = bottle, pads, contacts, hand_model
        self.bottle_index = np.array([0], dtype=np.uint32)
        self.pad_indices = np.arange(len(PAD_PATHS), dtype=np.uint32)
        self.pad_targets = np.zeros((len(PAD_PATHS), 7), dtype=np.float32)
        self.pad_targets[:, 6] = 1
        self.inertia = self.bottle.get_inertias()[0].reshape(3, 3).copy()

    def damp_rotation(self, orientation, angular_velocity, coefficient):
        if np.linalg.norm(angular_velocity) < 0.04:
            return
        rotation = Rotation.from_quat(orientation).as_matrix()
        torque = -coefficient * rotation @ self.inertia @ rotation.T @ angular_velocity
        self.bottle.apply_forces_and_torques_at_position(
            None, np.asarray([torque], dtype=np.float32), None, self.bottle_index, True,
        )

    def run(self, throw, step_physics, render=None, arm=None, log=False, trace=False):
        from pxr import Gf

        limited = isinstance(throw, LimitedThrow)
        if limited and arm is None:
            raise ValueError("a rate-limited trial requires the Tianji URDF")
        limits_report = throw.validate(arm[0]) if limited else None
        cap_pinch = limited and throw.cap_pinch
        bottle_start = throw.bottle_start if limited else START
        lift_at = throw.lift_at if limited else LIFT_AT
        release_at = throw.release_at if limited else throw.open_at
        # Clear contact manifolds before restoring the same tabletop start pose.
        clear = np.array([[0, 0, 1.6, 0, 0, 0, 1]], dtype=np.float32)
        zero = np.zeros((1, 6), dtype=np.float32)
        self.bottle.set_transforms(clear, self.bottle_index)
        self.bottle.set_velocities(zero, self.bottle_index)
        for _ in range(8):
            step_physics()
        pose = np.array([[*bottle_start, 0, 0, 0, 1]], dtype=np.float32)
        self.bottle.set_transforms(pose, self.bottle_index)
        self.bottle.set_velocities(zero, self.bottle_index)
        if arm is not None:
            rig, handles, hand = arm
            if not limited:
                angles = rig.initialize(throw.command(0)[0])
        lifted = False
        peak_grip = 0.0
        detach = impact = None
        pending_release = None
        no_contact_frames = 0
        release_confirmation_frames = 8 if limited and throw.forward_flick else 1
        last_pitch = turn = upright_s = 0.0
        apex = HALF
        last_finger_command = None
        flight_steps = 0
        post_release_force = 0.0
        peak_other_force = 0.0
        cap_grasped = False
        opening_command_at = None
        last_pinch_contact = None
        for frame in range(round((throw.open_at + 4.0) / DT)):
            t = frame * DT
            if limited:
                wrist, jacobian, links = rig.evaluate(throw.joints_at(frame))
                finger_command = throw.fingers_at(t)
            else:
                wrist, curl = throw.command(t)
                finger_command = (curl, None, 0.0)
            if finger_command != last_finger_command:
                fingers, local_pads = self.hand_model.evaluate(*finger_command)
                last_finger_command = finger_command
                if cap_pinch and t > release_at and opening_command_at is None:
                    opening_command_at = t
            if arm is not None:
                if not limited:
                    angles, wrist, links = rig.follow(wrist, angles)
                if hand is not None and frame % 4 == 0:
                    for name, transform in links.items():
                        if name in handles:
                            handles[name].Set(Gf.Matrix4d(*transform.T.reshape(-1).tolist()))
                    hand.set_joint_angles(fingers)

            self.pad_targets[:, :3] = local_pads @ wrist[:3, :3].T + wrist[:3, 3]
            if frame == 0:
                self.pads.set_transforms(self.pad_targets, self.pad_indices)
            self.pads.set_kinematic_targets(self.pad_targets, self.pad_indices)

            if detach is not None and impact is None and flight_steps:
                # The angular-rate curve approximates water redistribution with a
                # torque on a rigid bottle with constant mass properties.
                ratio = angular_rate(flight_steps * DT, 1.0, throw.inertia_growth) / angular_rate(
                    (flight_steps - 1) * DT, 1.0, throw.inertia_growth,
                )
                self.damp_rotation(position[3:], velocity[3:], (1 - ratio) / DT)
            elif impact is not None and t - impact["time_s"] < 0.32:
                # Transient drag from water moving relative to the bottle after
                # impact. The torque opposes angular speed in any orientation.
                self.damp_rotation(position[3:], velocity[3:], 24.0)
            step_physics()
            position = self.bottle.get_transforms()[0].copy()
            velocity = self.bottle.get_velocities()[0].copy()
            forces = np.linalg.norm(self.contacts.get_normal_contact_force_matrix(DT)[0], axis=1)
            grip, table = forces[:5].sum(), forces[5]
            rotation = Rotation.from_quat(position[3:]).as_matrix()
            if cap_pinch:
                peak_other_force = max(peak_other_force, float(forces[2:5].max()))
                if forces[:2].max() > 0.03 and opening_command_at is None:
                    last_pinch_contact = t
                if t < lift_at and not cap_grasped and forces[:2].min() > 0.10:
                    cap_levels = (self.pad_targets[:2, :3] - position[:3]) @ rotation[:, 2]
                    cap_grasped = bool(np.all(np.abs(cap_levels - CAP_CENTER_HEIGHT) < 0.009))
            if trace and ((frame % 24 == 0 and t <= throw.open_at + 0.4)
                          or release_at - 0.03 <= t <= throw.open_at + 0.16):
                print(f"[trace] {t:.4f} fingers {finger_command} {position[:3].round(4)}; "
                      f"forces {forces.round(2)}", flush=True)
            apex = max(apex, float(position[2]))
            peak_grip = max(peak_grip, float(grip))
            lifted |= position[2] > bottle_start[2] + 0.035 and grip > 0.10 and t >= lift_at
            up = rotation[:, 2]
            pitch = math.atan2(up[0], up[2])
            # Confirm a continuous gap so a one-step dip in normal force during
            # finger opening is not used as the start of ballistic flight.
            if (lifted and detach is None and t >= release_at and grip < 0.03
                    and (not cap_pinch or opening_command_at is not None)
                    and position[2] > bottle_start[2] + 0.04):
                if no_contact_frames == 0:
                    pending_release = {
                        "time_s": t,
                        "height_m": float(position[2]),
                        "pitch_deg": math.degrees(pitch),
                        "vx_m_s": float(velocity[0]),
                        "vy_m_s": float(velocity[1]),
                        "vz_m_s": float(velocity[2]),
                        "pitch_rate_rad_s": float(velocity[4]),
                    }
                    if limited:
                        joint_velocity = (throw.joints_at(frame) - throw.joints_at(frame - 1)) / DT
                        pending_release["wrist_pitch_rate_rad_s"] = float(jacobian[4] @ joint_velocity)
                    if cap_pinch and opening_command_at is not None:
                        pending_release["opening_command_at_s"] = opening_command_at
                        pending_release["contact_lost_at_s"] = t + DT
                        pending_release["release_delay_ms"] = (t + DT - opening_command_at) * 1000
                        pending_release["x_m"], pending_release["y_m"] = map(float, position[:2])
                no_contact_frames += 1
                if no_contact_frames >= release_confirmation_frames:
                    detach = pending_release
                    last_pitch = math.radians(detach["pitch_deg"])
                    flight_steps = no_contact_frames - 1
                    if log:
                        print("[full] Detach from finger pads", json.dumps(detach), flush=True)
            else:
                no_contact_frames = 0
            if detach is not None and impact is None:
                flight_steps += 1
                post_release_force = max(post_release_force, float(grip))
                turn += math.remainder(pitch - last_pitch, math.tau)
                last_pitch = pitch
                if table > 0.05:
                    impact = {
                        "time_s": t,
                        "turn_deg": math.degrees(turn),
                        "tilt_deg": math.degrees(math.acos(np.clip(up[2], -1, 1))),
                        "bottom_first": bool(up[2] > 0.2),
                        "in_target": bool(TARGET_X[0] <= position[0] <= TARGET_X[1] and abs(position[1]) <= 0.16),
                        "on_table": on_landing_table(position, cap_pinch),
                        "x_m": float(position[0]),
                        "y_m": float(position[1]),
                    }
            standing = (
                impact is not None and up[2] >= math.cos(math.radians(10))
                and (not limited or on_landing_table(position, cap_pinch))
                and abs(position[2] - HALF) < 0.013
                and np.linalg.norm(velocity[:3]) < 0.07
                and np.linalg.norm(np.cross(velocity[3:], up)) < 0.75
            )
            upright_s = upright_s + DT if standing else 0.0
            if render is not None and frame % 4 == 0 and not render(t):
                return None

        outcome = Outcome(
            "landed", impact["turn_deg"], impact["tilt_deg"], impact["bottom_first"],
            impact["on_table"] if limited else impact["in_target"], upright_s,
        ) if impact is not None else Outcome("no_release")
        result = score(outcome)
        if limited and result.success and post_release_force > 0.03:
            result = replace(result, success=False, reward=0., result="finger_recontact")
        if limited and throw.forward_flick and result.success:
            if detach["vx_m_s"] <= 0 or detach["vz_m_s"] <= 0 or impact["turn_deg"] >= 0:
                result = replace(result, success=False, reward=0., result="wrong_throw_direction")
        if cap_pinch and result.success:
            if not cap_grasped or peak_other_force > 0.03:
                result = replace(result, success=False, reward=0., result="not_a_cap_pinch")
            elif last_pinch_contact is None or release_at - last_pinch_contact > 4 * DT:
                result = replace(result, success=False, reward=0., result="slipped_before_opening")
        metrics = {
            "success": result.success, "reward": result.reward, "result": result.result,
            "lifted_by_contact": bool(lifted), "peak_pad_force_n": peak_grip, "apex_m": apex,
            "detach": detach, "impact": impact, "upright_s": upright_s,
            "parameters": throw.report_parameters() if limited else asdict(throw),
            "joint_limits": limits_report,
            "post_release_pad_force_n": post_release_force,
            "final_z_m": float(position[2]), "final_tilt_deg": math.degrees(math.acos(np.clip(up[2], -1, 1))),
            "final_speed_m_s": float(np.linalg.norm(velocity[:3])),
            "final_wobble_rad_s": float(np.linalg.norm(np.cross(velocity[3:], up))),
        }
        if cap_pinch:
            metrics["cap_pinch"] = {
                "grasped_before_lift": cap_grasped,
                "peak_other_pad_force_n": peak_other_force,
                "last_pinch_contact_before_open_s": last_pinch_contact,
            }
            metrics["final_xy_m"] = position[:2].tolist()
        return metrics


def play(app, args):
    import omni.physics.tensors as tensors
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    import omni.kit.viewport.utility as vp
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    candidates = ([LimitedThrow.load(path, DT) for path in args.limited_trajectory]
                  or [Throw(**json.loads(path.read_text())) for path in args.parameters or [DEMO_PARAMETERS]])
    nominal = candidates[0]
    limited = isinstance(nominal, LimitedThrow)
    arm = None
    if args.arm_assets is not None:
        base = np.eye(4)
        base[:3, 3] = nominal.base_translation if limited else [-0.20, 0.56, -0.78]
        base[:3, :3] = Rotation.from_euler("z", nominal.base_yaw_rad if limited else math.pi / 2).as_matrix()
        rig = Tianji(args.arm_assets / "urdf" / "Stand-m6.urdf", base)
        if limited:
            sdk_limits = args.arm_assets.parent / "marvin_sdk" / "CommonConfig" / "ccs_m6_40.MvKDCfg"
            if sdk_limits.is_file():
                rig.apply_m6_sdk_limits(sdk_limits)
            for candidate in candidates:
                if not np.allclose(candidate.base_translation, nominal.base_translation):
                    raise ValueError("one batch must use the same arm base placement")
                if not math.isclose(candidate.base_yaw_rad, nominal.base_yaw_rad):
                    raise ValueError("one batch must use the same arm base yaw")
                if candidate.cap_pinch != nominal.cap_pinch:
                    raise ValueError("one batch must use the same bottle geometry and hand grip")
                if (not math.isclose(candidate.bottle_mass_kg, nominal.bottle_mass_kg)
                        or not math.isclose(candidate.bottle_com_z_m, nominal.bottle_com_z_m)):
                    raise ValueError("one batch must use the same bottle mass and center of mass")
                if (max(candidate.bottle_start[2], nominal.bottle_start[2]) > HALF + 0.002
                        and not np.allclose(candidate.bottle_start, nominal.bottle_start)):
                    raise ValueError("one elevated batch must use the same pickup fixture")
                candidate.validate(rig)
        arm = rig, args.arm_assets
    forward_flick = limited and nominal.forward_flick
    cap_pinch = limited and nominal.cap_pinch
    handles, hand, camera = _stage(
        arm if not args.headless else None, nominal.bottle_start if limited else START,
        show_target=not limited, forward_flick=forward_flick,
        bottle_mass=nominal.bottle_mass_kg if limited else DEFAULT_BOTTLE_MASS_KG,
        bottle_com_z=nominal.bottle_com_z_m if limited else DEFAULT_BOTTLE_COM_Z_M,
        cap_pinch=cap_pinch,
    )
    if arm is not None:
        arm = rig, handles, hand
    SimulationManager.setup_simulation(dt=DT, device="cpu")
    SimulationManager.enable_fabric(False)
    RenderingManager.set_dt(1 / 60)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    SimulationManager.initialize_physics()
    view = tensors.create_simulation_view("numpy", stage_id=omni.usd.get_context().get_stage_id())
    view.set_subspace_roots("/")
    bodies = view.create_rigid_body_view("/World/Bottle")
    pads = view.create_rigid_body_view("/World/Pad_*")
    if list(pads.prim_paths) != PAD_PATHS:
        raise RuntimeError(f"unexpected PhysX pad ordering: {pads.prim_paths}")
    contacts = view.create_rigid_contact_view("/World/Bottle", [*PAD_PATHS, "/World/Table"])
    trial = ContactTrial(bodies, pads, contacts, Revo2CapPinch() if cap_pinch else Revo2Pads())
    if args.headless:
        if args.scan:
            rng = np.random.default_rng(11)
            candidates.extend(
                replace(nominal, height=rng.uniform(0.48, 0.60), rate=rng.uniform(15.5, 22),
                        pitch=rng.uniform(-0.34, 0.26), vx=rng.uniform(1.12, 1.62),
                        vz=rng.uniform(1.15, 1.75), opening=rng.uniform(0.045, 0.09),
                        inertia_growth=rng.uniform(0.35, 1.30))
                for _ in range(args.scan)
            )
        for index, throw in enumerate(candidates):
            result = trial.run(throw, SimulationManager.step, arm=arm, trace=args.trace and index == 0)
            print("[full]", index, json.dumps(result), flush=True)
        return

    viewport = vp.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = camera
    finger_release = limited and nominal.finger_release is not None
    show_slow_motion = finger_release or cap_pinch
    panel = ui.Window("Tianji + Revo2 | Contact Flip", width=420, height=196 if show_slow_motion else 170)
    panel.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
    replay = False

    def request_replay():
        nonlocal replay
        replay = True

    with panel.frame:
        with ui.VStack(spacing=5):
            stage_label = ui.Label("Loading", height=30)
            verdict = ui.Label("", height=35)
            ui.Button("Replay from table", clicked_fn=request_replay, height=32)
            if show_slow_motion:
                with ui.HStack(height=22, spacing=6):
                    slow_release = ui.CheckBox(width=20)
                    slow_release.model.set_value(True)
                    ui.Label("Slow-motion release")

    def render(time):
        lift_at = nominal.lift_at if limited else LIFT_AT
        release_at = nominal.release_at if limited else nominal.open_at
        if cap_pinch and nominal.lift_at <= time < nominal.open_at + 0.14:
            stage_label.text = ("Elbow and wrist flick up" if time < release_at else "Thumb and index open")
        elif finger_release and nominal.thumb_press_at <= time < nominal.open_at + 0.14:
            stage_label.text = ("Thumb presses / wrist loads" if time < release_at else
                                "Index lifts / wrist flicks up")
        else:
            lift_label = "Upper arm reaches forward" if forward_flick else "Tianji lifts and swings"
            grip_label = "Thumb and index pinch the cap" if cap_pinch else "Approach & close Revo2"
            stage_label.text = (grip_label if time < lift_at else
                                lift_label if time < release_at else "Flight and landing")
        RenderingManager.render()
        if (show_slow_motion and slow_release.model.as_bool
                and nominal.thumb_press_at - 0.045 <= time < nominal.open_at + 0.18):
            for _ in range(3):
                RenderingManager.render()
        return app.is_running() and not replay

    while app.is_running():
        replay = False
        verdict.text = "URDF joint-speed limits active" if limited else "Fingertip contact drives the bottle"
        result = trial.run(nominal, SimulationManager.step, render, arm, log=True)
        if result is None:
            continue
        print("[full] GUI", json.dumps(result), flush=True)
        turn = result["impact"]["turn_deg"] if result["impact"] else None
        verdict.text = (f"{abs(turn):.0f}°  |  upright {result['upright_s']:.2f} s"
                        if turn is not None else "Bottle did not reach the table after release")
        if cap_pinch and result["detach"] and "release_delay_ms" in result["detach"]:
            delay = result["detach"]["release_delay_ms"]
            verdict.text += f"  |  release {delay:.1f} ms"
        for _ in range(70):
            if not app.is_running() or replay:
                break
            RenderingManager.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-assets", type=Path, help="vendor Stand-m6 directory")
    parser.add_argument("--parameters", type=Path, action="append", default=[],
                        help="throw parameters in JSON, repeatable for headless verification")
    parser.add_argument("--limited-trajectory", type=Path, action="append", default=[],
                        help="validated joint-space samples in JSON, repeatable in headless mode")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scan", type=int, default=0, help="headless calibration trials")
    parser.add_argument("--trace", action="store_true", help="print the first calibration contact trace")
    args = parser.parse_args()
    if not args.headless and args.arm_assets is None:
        parser.error("--arm-assets is required for graphical playback")
    if args.limited_trajectory and (args.arm_assets is None or args.scan or args.parameters):
        parser.error("--limited-trajectory requires --arm-assets and is separate from --parameters/--scan")

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": args.headless, "width": 1380, "height": 860,
        "renderer": "MinimalRendering", "minimal_shading_mode": 4 if args.headless else 2,
        "multi_gpu": False, "sync_loads": True,
    })
    try:
        play(app, args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
