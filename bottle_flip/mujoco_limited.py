"""Cross-check rate-limited motion and bottle contact in the delivered MuJoCo model."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .data import Outcome
from .isaac_full_demo import DT, HALF, RADIUS, TABLE_CENTER_X, TABLE_HALF_SIZE, on_landing_table
from .limited_throw import DEFAULT_BOTTLE_MASS_KG, LimitedThrow
from .reward import score
from .robot_kinematics import PAD_RADIUS, Revo2Pads, Tianji
from .simulator import angular_rate


def _formatted(items):
    return " ".join(str(float(item)) for item in items)


def _contact_model(mujoco, scene, trajectory):
    model = ET.parse(scene).getroot()
    options = model.find("option")
    options.set("timestep", str(DT / 2))
    for asset in model.findall("asset/*"):
        if asset.get("file"):
            asset.set("file", str((scene.parent / asset.get("file")).resolve()))
    world = model.find("worldbody")
    base = world.find("body[@name='base_link']")
    base.set("pos", _formatted(trajectory.base_translation))
    half_yaw = trajectory.base_yaw_rad / 2
    base.set("quat", _formatted((math.cos(half_yaw), 0, 0, math.sin(half_yaw))))
    model.remove(model.find("keyframe"))

    def surface(name, position, size):
        ET.SubElement(world, "geom", name=name, type="box", pos=_formatted(position),
                      size=_formatted(size), contype="2", conaffinity="1",
                      solref="0.011 1", friction="0.85 0.004 0.0001")

    surface("landing_table", (TABLE_CENTER_X, 0, -TABLE_HALF_SIZE[2]), TABLE_HALF_SIZE)
    pick_height = trajectory.bottle_start[2] - HALF
    if pick_height > 0.002:
        surface("pickup_fixture", (*trajectory.bottle_start[:2], pick_height / 2),
                (0.07, 0.07, pick_height / 2))
    bottle = ET.SubElement(world, "body", name="bottle", pos=_formatted(trajectory.bottle_start))
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    inertia = np.array([0.00105, 0.00105, 0.000091]) * trajectory.bottle_mass_kg / DEFAULT_BOTTLE_MASS_KG
    ET.SubElement(bottle, "inertial", pos=_formatted((0, 0, trajectory.bottle_com_z_m)),
                  mass=str(trajectory.bottle_mass_kg), diaginertia=_formatted(inertia))
    ET.SubElement(bottle, "geom", name="bottle_hull", type="cylinder", size=f"{RADIUS} {HALF}",
                  friction="0.55 0.004 0.0001", contype="1", conaffinity="6")
    for index in range(5):
        pad = ET.SubElement(world, "body", name=f"pad_{index}", pos=f"0 0 {-1-index/20}")
        ET.SubElement(pad, "freejoint", name=f"pad_free_{index}")
        ET.SubElement(pad, "geom", name=f"tip_{index}", type="sphere", size=str(PAD_RADIUS),
                      mass="0.20", friction="2 0.004 0.0001", contype="4", conaffinity="1",
                      solref="0.004 1")
    return mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"))


def evaluate(scene, rig, trajectory):
    import mujoco

    limits = trajectory.validate(rig)
    model = _contact_model(mujoco, scene, trajectory)
    data = mujoco.MjData(model)
    jpos, jvel = model.jnt_qposadr, model.jnt_dofadr
    arm_index = [model.joint(joint.name).id for joint in rig.chain[:-1]]
    pad_index = [model.joint(f"pad_free_{index}").id for index in range(5)]
    tip_geoms = {model.geom(f"tip_{index}").id for index in range(5)}
    bottle_id = model.body("bottle").id
    bottle_geom = model.geom("bottle_hull").id
    table_geom = model.geom("landing_table").id
    tcp = model.body("TCP_Link_R").id
    pads = Revo2Pads()
    previous_q = trajectory.joints_at(0)
    previous_pads = None
    force = np.zeros(6)
    spatial = np.zeros(6)
    lifted = False
    detach = impact = None
    peak_before_open = float(trajectory.bottle_start[2])
    at_open = None
    last_pitch = turn = upright = fk_error = post_force = 0.
    dt = model.opt.timestep
    previous_finger_command = None
    for frame in range(round((trajectory.open_at + 4) / dt)):
        t = frame * dt
        index, fraction = divmod(frame, 2)
        q = (1 - fraction / 2) * trajectory.joints_at(index) + fraction / 2 * trajectory.joints_at(index + 1)
        data.qpos[jpos[arm_index]] = q
        data.qvel[jvel[arm_index]] = (q - previous_q) / dt
        wrist, *_ = rig.evaluate(q)
        previous_q = q
        finger_command = trajectory.fingers_at(t)
        if finger_command != previous_finger_command:
            _, local_pads = pads.evaluate(*finger_command)
            previous_finger_command = finger_command
        centers = local_pads @ wrist[:3, :3].T + wrist[:3, 3]
        tip_speed = np.zeros_like(centers) if previous_pads is None else (centers - previous_pads) / dt
        previous_pads = centers
        for index, joint in enumerate(pad_index):
            data.qpos[jpos[joint]:jpos[joint]+7] = [*centers[index], 1, 0, 0, 0]
            data.qvel[jvel[joint]:jvel[joint]+6] = [*tip_speed[index], 0, 0, 0]

        data.xfrc_applied[bottle_id] = 0
        if detach is not None:
            time_air = t - detach["time_s"]
            scale = 0
            if impact is None:
                rate = angular_rate(time_air, 1, trajectory.inertia_growth)
                scale = (1 - rate / angular_rate(max(0, time_air - dt), 1, trajectory.inertia_growth)) / dt
            elif t - impact["time_s"] < 0.32:
                scale = 24.
            if scale:
                axes = data.ximat[bottle_id].reshape(3, 3)
                data.xfrc_applied[bottle_id, 3:] = -scale * axes @ (
                    model.body_inertia[bottle_id] * (axes.T @ spatial[:3])
                )
        mujoco.mj_step(model, data)
        fk_error = max(fk_error, float(np.linalg.norm(data.xpos[tcp] - wrist[:3, 3])))
        position = data.xpos[bottle_id]
        up = data.xmat[bottle_id].reshape(3, 3)[:, 2]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bottle_id, spatial, 0)
        grip = table = 0.
        for address in range(data.ncon):
            pair = data.contact[address].geom
            if bottle_geom not in pair:
                continue
            other = pair[0] if pair[1] == bottle_geom else pair[1]
            if other in tip_geoms or other == table_geom:
                mujoco.mj_contactForce(model, data, address, force)
                if other in tip_geoms:
                    grip += force[0]
                else:
                    table += force[0]
        lifted |= position[2] > trajectory.bottle_start[2] + 0.035 and grip > 0.10 and t >= trajectory.lift_at
        if t < trajectory.open_at:
            peak_before_open = max(peak_before_open, float(position[2]))
        elif at_open is None:
            at_open = {"bottle_height_m": float(position[2]), "pad_force_n": float(grip)}
        pitch = math.atan2(up[0], up[2])
        if (lifted and detach is None and t >= trajectory.release_at and grip < 0.03
                and position[2] > trajectory.bottle_start[2] + 0.04):
            detach = {"time_s": t, "height_m": float(position[2]), "pitch_deg": math.degrees(pitch),
                      "pitch_rate_rad_s": float(spatial[1]), "vx_m_s": float(spatial[3]),
                      "vz_m_s": float(spatial[5])}
            last_pitch = pitch
        if detach is not None and impact is None:
            post_force = max(post_force, grip)
            turn += math.remainder(pitch - last_pitch, math.tau)
            last_pitch = pitch
            if table > 0.05:
                impact = {"time_s": t, "turn_deg": math.degrees(turn),
                          "tilt_deg": math.degrees(math.acos(np.clip(up[2], -1, 1))),
                          "bottom_first": bool(up[2] > 0.2),
                          "on_table": on_landing_table(position),
                          "x_m": float(position[0]), "y_m": float(position[1])}
        standing = (impact is not None and up[2] >= math.cos(math.radians(10))
                    and on_landing_table(position)
                    and abs(position[2] - HALF) < 0.013 and np.linalg.norm(spatial[3:]) < 0.07
                    and np.linalg.norm(np.cross(spatial[:3], up)) < 0.75)
        upright = upright + dt if standing else 0.

    outcome = Outcome("landed", impact["turn_deg"], impact["tilt_deg"], impact["bottom_first"],
                      impact["on_table"], upright) if impact is not None else Outcome("no_release")
    result = score(outcome)
    if result.success and post_force > 0.03:
        result = replace(result, success=False, reward=0., result="finger_recontact")
    if trajectory.forward_flick and result.success:
        if detach["vx_m_s"] <= 0 or detach["vz_m_s"] <= 0 or impact["turn_deg"] >= 0:
            result = replace(result, success=False, reward=0., result="wrong_throw_direction")
    return {"engine": f"MuJoCo {mujoco.__version__}", "arm_tracking": "joint_position_replay",
            "success": result.success, "result": result.result,
            "lifted_by_contact": bool(lifted), "detach": detach, "impact": impact, "upright_s": upright,
            "max_bottle_height_before_open_m": peak_before_open, "at_open": at_open,
            "post_release_pad_force_n": float(post_force), "max_tcp_fk_difference_m": fk_error,
            "joint_limits": limits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--arm-assets", type=Path, required=True)
    parser.add_argument("trajectory", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.trajectory:
        trajectory = LimitedThrow.load(path, DT)
        if trajectory.cap_pinch:
            parser.error(f"{path}: cap-pinch playback requires the Isaac Sim contact model")
        base = np.eye(4)
        base[:3, 3] = trajectory.base_translation
        base[:3, :3] = Rotation.from_euler("z", trajectory.base_yaw_rad).as_matrix()
        rig = Tianji(args.arm_assets / "urdf" / "Stand-m6.urdf", base)
        sdk_limits = args.arm_assets.parent / "marvin_sdk" / "CommonConfig" / "ccs_m6_40.MvKDCfg"
        if sdk_limits.is_file():
            rig.apply_m6_sdk_limits(sdk_limits)
        print(json.dumps({"trajectory": str(path), **evaluate(args.scene, rig, trajectory)}))


if __name__ == "__main__":
    main()
