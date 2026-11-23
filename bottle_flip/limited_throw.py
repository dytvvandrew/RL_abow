"""Preflight validation for rate-limited joint-space throws."""

import json
import math
from dataclasses import dataclass

import numpy as np

from .simulator import BOTTLE_HALF_LENGTH_M


WRIST_COUPLING_LIMIT = math.radians(110.5)
WRIST_COUPLING_SCALE = 1.025
THUMB_PRESS_RAMP_S = 0.09
DEFAULT_BOTTLE_MASS_KG = 0.177
DEFAULT_BOTTLE_COM_Z_M = -0.045


def smooth(value):
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class FingerRelease:
    index_lead_s: float
    thumb_press_rad: float


@dataclass(frozen=True)
class LimitedThrow:
    joints: np.ndarray
    base_translation: np.ndarray
    bottle_start: np.ndarray
    grasp_at: float
    lift_at: float
    open_at: float
    release_ramp: float
    dt: float
    inertia_growth: float
    finger_release: FingerRelease | None = None
    base_yaw_rad: float = math.pi / 2
    forward_flick: bool = False
    release_ramp_fraction: float = 0.15
    bottle_mass_kg: float = DEFAULT_BOTTLE_MASS_KG
    bottle_com_z_m: float = DEFAULT_BOTTLE_COM_Z_M
    cap_pinch: bool = False

    @classmethod
    def load(cls, path, dt):
        data = json.loads(path.read_text())
        joints = np.asarray(data["joints_rad"], dtype=float)
        base = np.asarray(data["base_translation_m"], dtype=float)
        start = np.asarray(data["bottle_start_m"], dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 7 or len(joints) < 2:
            raise ValueError(f"{path}: expected a sequence of seven-axis joint samples")
        if any(array.shape != (3,) or not np.isfinite(array).all() for array in (base, start)):
            raise ValueError(f"{path}: expected finite XYZ base and bottle positions")
        if data["dt_s"] != dt or not np.isfinite(joints).all():
            raise ValueError(f"{path}: invalid joint sample or physics-step mismatch")
        grasp, lift, opening, ramp, inertia = (
            float(data[key]) for key in ("grasp_at_s", "lift_at_s", "open_at_s",
                                         "release_ramp_s", "inertia_growth")
        )
        if not all(map(math.isfinite, (grasp, lift, opening, ramp, inertia))):
            raise ValueError(f"{path}: timing and inertia must be finite")
        valid_phases = 0 < grasp < lift - 0.20 and lift < opening < (len(joints) - 1) * dt
        if not valid_phases or ramp <= 0 or inertia < 0:
            raise ValueError(f"{path}: invalid grasp, lift or opening phases")
        release = None
        if "finger_release" in data:
            release = FingerRelease(**data["finger_release"])
            if not (0 <= release.index_lead_s <= 0.10
                    and 0 < release.thumb_press_rad <= 0.24
                    and opening - release.index_lead_s - THUMB_PRESS_RAMP_S > lift):
                raise ValueError(f"{path}: invalid finger release")
        base_yaw = float(data.get("base_yaw_rad", math.pi / 2))
        if not math.isfinite(base_yaw):
            raise ValueError(f"{path}: base yaw must be finite")
        forward = data.get("forward_flick", False)
        if not isinstance(forward, bool):
            raise ValueError(f"{path}: forward_flick must be a boolean")
        ramp_fraction = float(data.get("release_ramp_fraction", 0.15))
        if not 0 < ramp_fraction < 1:
            raise ValueError(f"{path}: release ramp fraction must be between zero and one")
        bottle_mass = float(data.get("bottle_mass_kg", DEFAULT_BOTTLE_MASS_KG))
        bottle_com_z = float(data.get("bottle_com_z_m", DEFAULT_BOTTLE_COM_Z_M))
        if not (math.isfinite(bottle_mass) and bottle_mass > 0
                and abs(bottle_com_z) < BOTTLE_HALF_LENGTH_M):
            raise ValueError(f"{path}: bottle mass must be positive and its center of mass must be inside the bottle")
        cap_pinch = data.get("cap_pinch", False)
        if not isinstance(cap_pinch, bool) or (cap_pinch and (not forward or release is not None)):
            raise ValueError(f"{path}: cap_pinch uses a forward flick with simultaneous finger opening")
        return cls(joints, base, start, grasp, lift, opening, ramp, dt, inertia, release,
                   base_yaw, forward, ramp_fraction, bottle_mass, bottle_com_z, cap_pinch)

    def validate(self, rig):
        margin = 1e-7
        if np.any(self.joints < rig.lower - margin) or np.any(self.joints > rig.upper + margin):
            raise ValueError("joint trajectory exceeds the URDF position limits")
        wrist = np.abs(self.joints[:, 6]) + WRIST_COUPLING_SCALE * np.abs(self.joints[:, 5])
        if wrist.max() > WRIST_COUPLING_LIMIT + margin:
            raise ValueError("joint trajectory exceeds the J6/J7 coupled position limit")
        peak = np.max(np.abs(np.diff(self.joints, axis=0)), axis=0) / self.dt
        if np.any(peak > rig.speed_limits + margin):
            raise ValueError("joint trajectory exceeds the URDF velocity limits")
        report = {
            "peak_rad_s": peak.tolist(),
            "limit_rad_s": rig.speed_limits.tolist(),
            "peak_wrist_coupling_deg": math.degrees(float(wrist.max())),
        }
        if self.forward_flick:
            if rig.forward_axis[0] < math.cos(math.radians(2.0)):
                raise ValueError("the Stand-m6 factory front (-Y) must face world +X toward the table")
            upper = rig.upper_arm_vectors(self.joints)
            forward_angle = np.rad2deg(np.arctan2(upper[:, 0], -upper[:, 2]))
            side_mm = 1000 * float(np.max(np.abs(upper[:, 1])))
            if (abs(forward_angle[0]) > 2 or forward_angle.min() < -0.1
                    or np.any(np.diff(forward_angle) < -1e-5) or side_mm > 5):
                raise ValueError("upper arm must start vertical and move forward in the throwing plane")
            report["upper_arm"] = {
                "initial_forward_deg": float(forward_angle[0]),
                "max_forward_deg": float(forward_angle.max()),
                "max_side_mm": side_mm,
            }
        return report

    def joints_at(self, frame):
        return self.joints[min(frame, len(self.joints) - 1)]

    @property
    def release_at(self):
        return self.open_at - self.finger_release.index_lead_s if self.finger_release else self.open_at

    @property
    def thumb_press_at(self):
        return self.release_at - THUMB_PRESS_RAMP_S if self.finger_release else self.release_at

    def curl_at(self, time):
        if time < self.open_at:
            return smooth((time - self.grasp_at) / (self.lift_at - self.grasp_at - 0.20))
        elapsed = time - self.open_at
        if self.cap_pinch:
            return 1.0 - smooth(elapsed / self.release_ramp)
        if elapsed < self.release_ramp:
            return 1.0 - self.release_ramp_fraction * smooth(elapsed / self.release_ramp)
        return (1.0 - self.release_ramp_fraction) * (1.0 - smooth((elapsed - self.release_ramp) / 0.52))

    def fingers_at(self, time):
        curl = self.curl_at(time)
        if self.finger_release is None or time < self.thumb_press_at:
            return curl, None, 0.0
        index = curl if time < self.release_at else self.curl_at(time + self.finger_release.index_lead_s)
        press = smooth((time - self.thumb_press_at) / THUMB_PRESS_RAMP_S)
        press *= 1.0 - smooth((time - self.open_at) / 0.085)
        return curl, index, self.finger_release.thumb_press_rad * press

    def report_parameters(self):
        parameters = {
            "trajectory": "joint_space_rate_limited",
            "grasp_at_s": self.grasp_at,
            "lift_at_s": self.lift_at,
            "open_at_s": self.open_at,
            "release_ramp_s": self.release_ramp,
            "base_translation_m": self.base_translation.tolist(),
            "base_yaw_rad": self.base_yaw_rad,
            "bottle_start_m": self.bottle_start.tolist(),
            "bottle_mass_kg": self.bottle_mass_kg,
            "bottle_com_z_m": self.bottle_com_z_m,
            "inertia_growth": self.inertia_growth,
        }
        if self.finger_release is not None:
            parameters["finger_release"] = {
                "thumb_press_at_s": self.thumb_press_at,
                "index_lift_at_s": self.release_at,
                "index_lead_s": self.finger_release.index_lead_s,
                "thumb_press_rad": self.finger_release.thumb_press_rad,
            }
        if self.forward_flick:
            parameters["forward_flick"] = True
        if self.cap_pinch:
            parameters["cap_pinch"] = True
        if self.release_ramp_fraction != 0.15:
            parameters["release_ramp_fraction"] = self.release_ramp_fraction
        return parameters
