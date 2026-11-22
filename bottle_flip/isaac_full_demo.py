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
from .revo2_preview import Revo2Preview, add_stl
from .reward import score
from .robot_kinematics import GRIP_CENTER, PAD_RADIUS, Revo2Pads, Tianji, origin
from .simulator import BOTTLE_HALF_LENGTH_M as HALF, BOTTLE_RADIUS_M as RADIUS, angular_rate


DT = 1 / 240
LIFT_AT = 0.90
START = np.array([-0.10, 0.0, HALF])
PALM_MAP = Rotation.from_euler("z", 0.91).as_matrix() @ Rotation.from_euler("x", math.pi / 2).as_matrix()
TARGET_X = (0.105, 0.62)
PAD_PATHS = [f"/World/Pad_{index}" for index in range(5)]
DEMO_PARAMETERS = Path(__file__).resolve().parents[1] / "assets" / "isaac_contact_flip.json"


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


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


def _stage(arm):
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
    cube("Table", (0.14, 0, -0.033), (1.05, 0.63, 0.066), (0.38, 0.46, 0.51), table_mat)
    cube("TableBase", (0.15, 0, -0.46), (0.52, 0.40, 0.80), (0.20, 0.25, 0.29))
    cube("Target", (sum(TARGET_X) / 2, 0, 0.0003),
         (TARGET_X[1] - TARGET_X[0], 0.32, 0.0006), (0.07, 0.48, 0.34))

    bottle = UsdGeom.Xform.Define(stage, "/World/Bottle")
    bottle.AddTranslateOp().Set(Gf.Vec3d(*START.tolist()))
    rigid = UsdPhysics.RigidBodyAPI.Apply(bottle.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(bottle.GetPrim())
    mass.CreateMassAttr(0.177)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, -0.045))
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
        for name in ["base_link"] + [joint.child for joint in rig.chain[:-1]]:
            link = UsdGeom.Xform.Define(stage, f"/World/Tianji/{name}")
            handles[name] = link.AddTransformOp()
            element = rig.urdf.find(f"link[@name='{name}']")
            for index, visual in enumerate(element.findall("visual")):
                filename = Path(visual.find("geometry/mesh").attrib["filename"]).name
                geom = add_stl(stage, f"{link.GetPath()}/Visual_{index}", assets / "meshes" / filename,
                               (0.56, 0.64, 0.70))
                pose = origin(visual.find("origin"))
                geom.AddTransformOp().Set(Gf.Matrix4d(*pose.T.reshape(-1).tolist()))
        handles["base_link"].Set(Gf.Matrix4d(*rig.base.T.reshape(-1).tolist()))
        mount = UsdGeom.Xform.Define(stage, "/World/Tianji/Link7_R/TCP_Link_R")
        mount.AddTransformOp().Set(Gf.Matrix4d(*rig.chain[-1].pose.T.reshape(-1).tolist()))
        hand = Revo2Preview(stage, f"{mount.GetPath()}/Revo2")

    UsdLux.DomeLight.Define(stage, "/World/Fill").CreateIntensityAttr(480.0)
    light = UsdLux.DistantLight.Define(stage, "/World/Key")
    light.CreateIntensityAttr(1050.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(38, -23, -15))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(24)
    eye, target = Gf.Vec3d(1.32, -1.70, 0.93), Gf.Vec3d(0.11, 0.19, 0.22)
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

        # Clear contact manifolds before restoring the same tabletop start pose.
        clear = np.array([[0, 0, 1.6, 0, 0, 0, 1]], dtype=np.float32)
        zero = np.zeros((1, 6), dtype=np.float32)
        self.bottle.set_transforms(clear, self.bottle_index)
        self.bottle.set_velocities(zero, self.bottle_index)
        for _ in range(8):
            step_physics()
        pose = np.array([[*START, 0, 0, 0, 1]], dtype=np.float32)
        self.bottle.set_transforms(pose, self.bottle_index)
        self.bottle.set_velocities(zero, self.bottle_index)
        if arm is not None:
            rig, handles, hand = arm
            angles = rig.initialize(throw.command(0)[0])
        lifted = False
        peak_grip = 0.0
        detach = impact = None
        last_pitch = turn = upright_s = 0.0
        apex = HALF
        last_curl = None
        flight_steps = 0
        post_release_force = 0.0
        for frame in range(round((throw.open_at + 4.0) / DT)):
            t = frame * DT
            wrist, curl = throw.command(t)
            if curl != last_curl:
                fingers, local_pads = self.hand_model.evaluate(curl)
                last_curl = curl
            if arm is not None:
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
                # Rising effective inertia reduces airborne spin. Flight translation
                # and the initial linear and angular release velocities come from PhysX.
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
            if trace and frame % 24 == 0 and t <= throw.open_at + 0.4:
                print(f"[trace] {t:.2f} curl {curl:.3f} {position[:3].round(4)}; "
                      f"forces {forces.round(2)}", flush=True)
            apex = max(apex, float(position[2]))
            peak_grip = max(peak_grip, float(grip))
            lifted |= position[2] > HALF + 0.035 and grip > 0.10 and t >= LIFT_AT
            up = Rotation.from_quat(position[3:]).as_matrix()[:, 2]
            pitch = math.atan2(up[0], up[2])
            if lifted and detach is None and t >= throw.open_at and grip < 0.03 and position[2] > HALF + 0.04:
                detach = {
                    "time_s": t,
                    "height_m": float(position[2]),
                    "pitch_deg": math.degrees(pitch),
                    "vx_m_s": float(velocity[0]),
                    "vy_m_s": float(velocity[1]),
                    "vz_m_s": float(velocity[2]),
                    "pitch_rate_rad_s": float(velocity[4]),
                }
                last_pitch = pitch
                if log:
                    print("[full] Detach from finger pads", json.dumps(detach), flush=True)
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
                        "x_m": float(position[0]),
                        "y_m": float(position[1]),
                    }
            standing = (
                impact is not None and up[2] >= math.cos(math.radians(10))
                and abs(position[2] - HALF) < 0.013
                and np.linalg.norm(velocity[:3]) < 0.07
                and np.linalg.norm(np.cross(velocity[3:], up)) < 0.75
            )
            upright_s = upright_s + DT if standing else 0.0
            if render is not None and frame % 4 == 0 and not render(t):
                return None

        outcome = Outcome(
            "landed", impact["turn_deg"], impact["tilt_deg"], impact["bottom_first"], impact["in_target"], upright_s,
        ) if impact is not None else Outcome("no_release")
        result = score(outcome)
        return {
            "success": result.success, "reward": result.reward, "result": result.result,
            "lifted_by_contact": bool(lifted), "peak_pad_force_n": peak_grip, "apex_m": apex,
            "detach": detach, "impact": impact, "upright_s": upright_s, "parameters": asdict(throw),
            "post_release_pad_force_n": post_release_force,
            "final_z_m": float(position[2]), "final_tilt_deg": math.degrees(math.acos(np.clip(up[2], -1, 1))),
            "final_speed_m_s": float(np.linalg.norm(velocity[:3])),
            "final_wobble_rad_s": float(np.linalg.norm(np.cross(velocity[3:], up))),
        }


def play(app, args):
    import omni.physics.tensors as tensors
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    import omni.kit.viewport.utility as vp
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    arm = None
    if args.arm_assets is not None:
        base = np.eye(4)
        base[:3, 3] = [-0.20, 0.56, -0.78]
        base[:3, :3] = Rotation.from_euler("z", math.pi / 2).as_matrix()
        rig = Tianji(args.arm_assets / "urdf" / "Stand-m6.urdf", base)
        arm = rig, args.arm_assets
    handles, hand, camera = _stage(arm if not args.headless else None)
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
    trial = ContactTrial(bodies, pads, contacts, Revo2Pads())
    candidates = [Throw(**json.loads(path.read_text())) for path in args.parameters or [DEMO_PARAMETERS]]
    nominal = candidates[0]
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
    panel = ui.Window("Tianji + Revo2 | Contact Flip", width=420, height=170)
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

    def render(time):
        stage_label.text = ("Approach & close Revo2" if time < LIFT_AT else
                            "Tianji lifts and swings" if time < nominal.open_at else
                            "Open fingers / PhysX flight and landing")
        RenderingManager.render()
        return app.is_running() and not replay

    while app.is_running():
        replay = False
        verdict.text = "Fingertip contact drives the bottle"
        result = trial.run(nominal, SimulationManager.step, render, arm, log=True)
        if result is None:
            continue
        print("[full] GUI", json.dumps(result), flush=True)
        turn = result["impact"]["turn_deg"] if result["impact"] else None
        verdict.text = (f"{abs(turn):.0f}°  |  upright {result['upright_s']:.2f} s"
                        if turn is not None else "Bottle did not reach the table after release")
        for _ in range(70):
            if not app.is_running() or replay:
                break
            RenderingManager.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-assets", type=Path, help="vendor Stand-m6 directory")
    parser.add_argument("--parameters", type=Path, action="append", default=[],
                        help="throw parameters in JSON, repeatable for headless verification")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--scan", type=int, default=0, help="headless calibration trials")
    parser.add_argument("--trace", action="store_true", help="print the first calibration contact trace")
    args = parser.parse_args()
    if not args.headless and args.arm_assets is None:
        parser.error("--arm-assets is required for graphical playback")

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
