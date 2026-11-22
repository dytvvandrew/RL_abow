"""Play IQL and fixed-policy launches using Isaac Sim 6.2 and PhysX."""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from .iql import UnsupportedStateError
from .policy import load_policies
from .revo2_preview import Revo2Preview
from .simulator import (
    BOTTLE_HALF_LENGTH_M, BOTTLE_RADIUS_M, GRAVITY_M_S2, TARGET_CENTER_M,
    TARGET_HALF_WIDTH_M, LaunchState, ThrowPlan, angular_rate, plan_throw,
)


DT = 1 / 120
GRASP_END = 0.72
DECISION_AT = 1.28
LANES = (
    ("iql", "IQL_Bottle", -0.62, (0.10, 0.67, 0.48)),
    ("fixed", "Fixed_Bottle", 0.12, (0.88, 0.39, 0.14)),
)


def build_scene(stage):
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics, UsdShade

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(GRAVITY_M_S2)
    material = UsdShade.Material.Define(stage, "/World/Contact")
    contact = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    contact.CreateStaticFrictionAttr(0.85)
    contact.CreateDynamicFrictionAttr(0.65)
    contact.CreateRestitutionAttr(0.02)

    def collider(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")

    def cube(name, position, size, color, collides=False):
        geometry = UsdGeom.Cube.Define(stage, f"/World/{name}")
        geometry.CreateSizeAttr(1.0)
        geometry.AddTranslateOp().Set(Gf.Vec3d(*position))
        geometry.AddScaleOp().Set(Gf.Vec3f(*size))
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if collides:
            collider(geometry.GetPrim())

    def cylinder(path, radius, height, z, color):
        geometry = UsdGeom.Cylinder.Define(stage, path)
        geometry.CreateAxisAttr(UsdGeom.Tokens.z)
        geometry.CreateRadiusAttr(radius)
        geometry.CreateHeightAttr(height)
        geometry.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        geometry.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return geometry

    cube("Floor", (0, 0, -0.62), (4.0, 4.0, 0.06), (0.16, 0.20, 0.22), True)
    for key, name, origin, color in LANES:
        cube(f"{key}_table", (origin + 0.19, 0, -0.035), (0.70, 0.50, 0.07), (0.38, 0.44, 0.46), True)
        cube(f"{key}_target", (origin + TARGET_CENTER_M, 0, 0.001),
             (2 * TARGET_HALF_WIDTH_M, 0.32, 0.002), color)
        cube(f"{key}_pedestal", (origin + 0.19, 0, -0.32), (0.29, 0.31, 0.52), (0.26, 0.31, 0.33))
        path = f"/World/{name}"
        bottle = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(bottle)
        UsdPhysics.MassAPI.Apply(bottle).CreateMassAttr(0.177)
        hull = cylinder(f"{path}/Hull", BOTTLE_RADIUS_M, 2 * BOTTLE_HALF_LENGTH_M, 0, color)
        collider(hull.GetPrim())
        hull.MakeInvisible()
        cylinder(f"{path}/LowerThird", BOTTLE_RADIUS_M, 0.070, -0.070, color)
        cylinder(f"{path}/UpperBody", 0.0315, 0.091, 0.0105, (0.66, 0.78, 0.82))
        cylinder(f"{path}/Shoulder", 0.023, 0.026, 0.069, (0.66, 0.78, 0.82))
        cylinder(f"{path}/Cap", 0.017, 0.023, 0.0935, color)

    UsdLux.DomeLight.Define(stage, "/World/Fill").CreateIntensityAttr(480.0)
    light = UsdLux.DistantLight.Define(stage, "/World/Key")
    light.CreateIntensityAttr(2400.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(40, -28, -15))
    camera = UsdGeom.Camera.Define(stage, "/World/ComparisonCamera")
    camera.CreateFocalLengthAttr(25.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 30))
    eye, target = Gf.Vec3d(0.49, -1.90, 1.21), Gf.Vec3d(0.01, 0, 0.23)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    return camera.GetPath()


def _ease(progress):
    fraction = min(1.0, max(0.0, progress))
    return fraction * fraction * (3 - 2 * fraction)


def _swing(start, end, end_velocity, progress):
    return (
        start + (end - start) * _ease(progress)
        + end_velocity * (DECISION_AT - GRASP_END) * progress * progress * (progress - 1)
    )


@dataclass
class GraspPlayback:
    origin: float
    plan: ThrowPlan

    @property
    def detach_at(self):
        return DECISION_AT + self.plan.launch.delay_s

    def bottle_pose(self, time):
        launch = self.plan.launch
        decision = self.plan.release_state
        start_x = self.origin - 0.09
        if time < GRASP_END:
            return (start_x, 0, BOTTLE_HALF_LENGTH_M), 0.0
        if time < DECISION_AT:
            progress = (time - GRASP_END) / (DECISION_AT - GRASP_END)
            x = _swing(start_x, self.origin - launch.vx_m_s * launch.delay_s, launch.vx_m_s, progress)
            z = _swing(BOTTLE_HALF_LENGTH_M, decision.height_m, decision.vz_m_s, progress)
            pitch = _swing(0.0, decision.pitch_rad, decision.pitch_rate_rad_s, progress)
            return (x, 0, z), pitch
        t = min(launch.delay_s, time - DECISION_AT)
        x = self.origin + launch.vx_m_s * (t - launch.delay_s)
        z = decision.height_m + decision.vz_m_s * t + 0.6 * t * t
        pitch = decision.pitch_rad + decision.pitch_rate_rad_s * t + 19 * t * t
        return (x, 0, z), pitch

    def move_hand(self, hand, time):
        center, pitch = self.bottle_pose(time)
        if time < GRASP_END:
            approach = 1 - _ease(time / 0.33)
            curl = _ease((time - 0.27) / 0.37)
            hand.pose(center, pitch, curl, (-0.10 * approach, 0.13 * approach, 0.12 * approach))
        else:
            after = time - self.detach_at
            opened = _ease((after + 0.06) / 0.34)
            withdrawn = _ease(after / 0.42)
            hand.pose(center, pitch + 0.22 * withdrawn, 1 - opened,
                      (-0.17 * withdrawn, 0.18 * withdrawn, 0.055 * withdrawn))


@dataclass
class Landing:
    last_pitch: float
    turn: float = 0.0
    impact_turn: float | None = None
    upright_s: float = 0.0

    def observe(self, position, orientation, velocity, spin):
        w, x, y, z = orientation
        up_x = 2 * (x * z + w * y)
        up_z = min(1.0, max(-1.0, 1 - 2 * (x * x + y * y)))
        pitch = math.atan2(up_x, up_z)
        self.turn += math.remainder(pitch - self.last_pitch, math.tau)
        self.last_pitch = pitch
        extent = BOTTLE_HALF_LENGTH_M * abs(up_z) + BOTTLE_RADIUS_M * math.sqrt(1 - up_z * up_z)
        if self.impact_turn is None and position[2] - extent <= 0.004:
            self.impact_turn = math.degrees(self.turn)
        standing = (
            self.impact_turn is not None and up_z > math.cos(math.radians(10))
            and abs(position[2] - BOTTLE_HALF_LENGTH_M) < 0.015
            and math.hypot(*velocity) < 0.08 and math.hypot(*spin) < 0.65
        )
        self.upright_s = self.upright_s + DT if standing else 0.0


def play(app, policies, seed, rounds):
    import omni.timeline
    import omni.ui as ui
    import omni.usd
    import omni.kit.viewport.utility as viewport_utils
    from isaacsim.core.experimental.prims import RigidPrim
    from isaacsim.core.experimental.utils.backend import use_backend
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    context = omni.usd.get_context()
    if context.get_stage() is None:
        context.new_stage()
    camera_path = build_scene(context.get_stage())
    hands = [
        Revo2Preview(context.get_stage(), f"/World/{key}_Revo2")
        for key, _, _, _ in LANES
    ]
    print(f"[isaac] Loaded {len(hands)} Revo2 hands with {len(hands[0].joints)} animated joints each", flush=True)
    SimulationManager.setup_simulation(dt=DT, device="cpu")
    SimulationManager.enable_fabric(False)
    RenderingManager.set_dt(DT)
    paths = [f"/World/{name}" for _, name, _, _ in LANES]
    bodies = RigidPrim(paths)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    SimulationManager.initialize_physics()
    RenderingManager.render()
    viewport = viewport_utils.get_active_viewport()
    if viewport is not None:
        viewport.camera_path = camera_path

    requested = None

    def request(advance):
        nonlocal requested
        requested = seed + advance

    panel = ui.Window("Bottle Flip | Isaac Sim", width=395, height=165)
    panel.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
    with panel.frame:
        with ui.VStack(spacing=4):
            summary = ui.Label("Preparing PhysX", height=28)
            lines = [ui.Label("", height=24), ui.Label("", height=24)]
            with ui.HStack(height=32, spacing=7):
                ui.Button("Replay this throw", clicked_fn=lambda: request(0))
                ui.Button("Next seed", clicked_fn=lambda: request(1))

    completed = 0
    with use_backend("tensor", raise_on_unsupported=True, raise_on_fallback=True):
        while app.is_running() and (rounds == 0 or completed < rounds):
            requested = None
            launches: list[LaunchState | None] = []
            playback = []
            released_at = [None] * len(LANES)
            landings = []
            for index, (key, _, origin, _) in enumerate(LANES):
                try:
                    plan = plan_throw(policies[key], seed)
                except UnsupportedStateError:
                    plan = None
                launch = plan.launch if plan is not None else None
                launches.append(launch)
                playback.append(GraspPlayback(origin, plan) if launch is not None else None)
                angle = launch.pitch_rad if launch else 0.0
                landings.append(Landing(angle))
                if launch is None:
                    bodies.set_world_poses(positions=[[origin, 0, -1.5]], indices=[index])
                    bodies.set_velocities(linear_velocities=[[0, 0, 0]],
                                          angular_velocities=[[0, 0, 0]], indices=[index])
                    hands[index].pose((origin, 0, -1.5), 0.0, 0.0)
                action = f"swing {plan.swing_id} | release {plan.release_id}" if launch else "aborted"
                lines[index].text = f"{key.upper()}   {action}"
            print(f"[isaac] Revo2 grasp and swing for seed {seed}", flush=True)
            summary.text = f"Seed {seed}   ·   left IQL   ·   right fixed"
            for step in range(round((DECISION_AT + 3.3) / DT)):
                if not app.is_running() or requested is not None:
                    break
                time = step * DT
                for index, launch in enumerate(launches):
                    if launch is None:
                        continue
                    motion = playback[index]
                    if time <= motion.detach_at + 0.43:
                        motion.move_hand(hands[index], time)
                    if released_at[index] is None:
                        position, pitch = motion.bottle_pose(time)
                        bodies.set_world_poses(
                            positions=[position],
                            orientations=[[math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]],
                            indices=[index],
                        )
                        if time < motion.detach_at:
                            bodies.set_velocities(linear_velocities=[[0, 0, 0]],
                                                  angular_velocities=[[0, 0, 0]], indices=[index])
                        else:
                            bodies.set_velocities(linear_velocities=[[launch.vx_m_s, 0, launch.vz_m_s]],
                                                  angular_velocities=[[0, launch.pitch_rate_rad_s, 0]],
                                                  indices=[index])
                            released_at[index] = time
                            print(f"[isaac] {LANES[index][0]} Revo2 opening; PhysX release at {time:.3f} s",
                                  flush=True)
                    elif landings[index].impact_turn is None:
                        rate = angular_rate(time - released_at[index], launch.pitch_rate_rad_s,
                                            launch.inertia_growth)
                        bodies.set_velocities(angular_velocities=[[0, rate, 0]], indices=[index])
                SimulationManager.step()
                positions, orientations = (array.numpy() for array in bodies.get_world_poses())
                velocities, spins = (array.numpy() for array in bodies.get_velocities())
                for index, released in enumerate(released_at):
                    if released is not None:
                        landings[index].observe(positions[index], orientations[index], velocities[index], spins[index])
                if step % 2 == 0:
                    RenderingManager.render()
            completed += 1
            if requested is not None:
                seed = requested
                continue
            for index, (key, _, origin, _) in enumerate(LANES):
                landing = landings[index]
                if landing.impact_turn is None:
                    continue
                upright = landing.upright_s >= 2
                in_zone = abs(positions[index][0] - origin - TARGET_CENTER_M) <= TARGET_HALF_WIDTH_M
                lines[index].text = (
                    f"{key.upper()}   {abs(landing.impact_turn):.0f}°   |   "
                    f"{'stood ≥2 s' if upright else 'did not stand ≥2 s'}"
                    f"{'' if in_zone else '   |   outside target'}"
                )
                print(f"[isaac] {lines[index].text}", flush=True)
            if rounds:
                continue
            for _ in range(90):
                if not app.is_running() or requested is not None:
                    break
                RenderingManager.render()
            if requested is not None:
                seed = requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("runs/sim_model.json"))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=0, help="0 keeps the graphical playback running")
    args = parser.parse_args()
    policies = load_policies(args.model)

    from isaacsim import SimulationApp

    app = SimulationApp({
        "headless": False, "width": 1280, "height": 820,
        "renderer": "MinimalRendering", "minimal_shading_mode": 2,
        "multi_gpu": False, "sync_loads": True,
    })
    try:
        play(app, policies, args.seed, args.rounds)
    finally:
        app.close()


if __name__ == "__main__":
    main()
