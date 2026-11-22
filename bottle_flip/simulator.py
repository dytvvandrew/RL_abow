"""Planar flight and approximate landing for closed-loop software checks."""

import math
import random
from dataclasses import dataclass

from .baseline import FixedPolicy
from .data import RELEASE_IDS, SWING_IDS, Outcome, ReleaseState, SwingState, Trial, swing_levels
from .iql import UnsupportedStateError


GRAVITY_M_S2 = 9.81
BOTTLE_HALF_LENGTH_M = 0.105
BOTTLE_RADIUS_M = 0.032
TARGET_CENTER_M = 0.27
TARGET_HALF_WIDTH_M = 0.13
FLIGHT_STEP_S = 0.002


@dataclass(frozen=True)
class LaunchState:
    delay_s: float
    z_m: float
    pitch_rad: float
    vx_m_s: float
    vz_m_s: float
    pitch_rate_rad_s: float
    inertia_growth: float


@dataclass(frozen=True)
class ThrowPlan:
    swing_state: SwingState
    swing_id: int
    release_state: ReleaseState | None
    release_id: int | None
    launch: LaunchState | None


def _clearance(height, pitch):
    extent = BOTTLE_HALF_LENGTH_M * abs(math.cos(pitch)) + BOTTLE_RADIUS_M * abs(math.sin(pitch))
    return height - extent


def angular_rate(t, initial_rate, inertia_growth):
    # Water relocation is represented by a time-varying inertia, not a fluid solver.
    fraction = 1 - math.exp(-(t / 0.25) ** 3)
    return initial_rate / (1 + inertia_growth * fraction)


def _fly(height, vz, vx, pitch, rate, inertia_growth):
    initial_pitch = pitch
    previous_clearance = _clearance(height, pitch)
    for step in range(1, 1001):
        t = step * FLIGHT_STEP_S
        previous_pitch = pitch
        pitch += FLIGHT_STEP_S * angular_rate(t - FLIGHT_STEP_S / 2, rate, inertia_growth)
        z = height + vz * t - 0.5 * GRAVITY_M_S2 * t * t
        clearance = _clearance(z, pitch)
        if clearance <= 0:
            fraction = previous_clearance / (previous_clearance - clearance)
            t -= (1 - fraction) * FLIGHT_STEP_S
            pitch = previous_pitch + fraction * (pitch - previous_pitch)
            break
        previous_clearance = clearance
    else:
        raise ValueError("the nominal simulated flight exceeded 2 s")

    tilt = math.acos(math.cos(pitch))
    bottom_first = math.cos(pitch) > 0.2
    end_sign = 1 if bottom_first else -1
    contact_x = vx * t - end_sign * BOTTLE_HALF_LENGTH_M * math.sin(pitch)
    in_target = abs(contact_x - TARGET_CENTER_M) <= TARGET_HALF_WIDTH_M
    impact_rate = abs(angular_rate(t, rate, inertia_growth))
    # Empirical stability envelope for the toy contact model.
    stable = bottom_first and (tilt / math.radians(18)) ** 2 + (impact_rate / 14.5) ** 2 <= 1
    return (
        Outcome("landed", math.degrees(pitch - initial_pitch), math.degrees(tilt),
                bottom_first, in_target, 2.0 if stable else 0.0),
        {"flight_duration_s": t, "first_contact_x_m": contact_x, "impact_rate_rad_s": impact_rate},
    )


def plan_throw(policy, seed):
    """Query both decisions using the same pre-release observations across backends."""
    rng = random.Random(seed)
    swing_state = SwingState(rng.gauss(0.42, 0.012), rng.gauss(0.0, 0.045))
    # Draw the same disturbances for both policies when they share a rollout seed.
    phase_noise = rng.gauss(0, 0.008)
    height_noise = rng.gauss(0, 0.012)
    pitch_noise = rng.gauss(0, 0.04)
    lift_noise = rng.gauss(0, 0.10)
    rate_noise = rng.gauss(0, 0.85)
    release_noise = rng.gauss(0, 0.003)
    horizontal_noise = rng.gauss(0, 0.080)
    inertia_growth = rng.uniform(1.05, 1.25)
    swing_id = policy.choose_swing(swing_state)
    lift_level, pitch_level = swing_levels(swing_id)
    release_state = ReleaseState(
        0.66 + phase_noise,
        swing_state.height_m + 0.04 + height_noise,
        -1.08 + swing_state.pitch_rad + pitch_noise,
        1.17 + 0.13 * lift_level + lift_noise,
        16.3 + 1.25 * pitch_level + rate_noise,
    )
    try:
        release_id = policy.choose_release(release_state, swing_id)
    except UnsupportedStateError:
        return ThrowPlan(swing_state, swing_id, None, None, None)

    delay = 0.065 + 0.012 * (release_id - 1) + release_noise
    launch = LaunchState(
        delay,
        release_state.height_m + release_state.vz_m_s * delay + 0.6 * delay * delay,
        release_state.pitch_rad + release_state.pitch_rate_rad_s * delay + 19 * delay * delay,
        0.60 + 0.04 * lift_level + horizontal_noise,
        release_state.vz_m_s + 1.2 * delay,
        release_state.pitch_rate_rad_s + 38 * delay,
        inertia_growth,
    )
    return ThrowPlan(swing_state, swing_id, release_state, release_id, launch)


def simulate_trial(policy, seed, trial_id, session_id):
    """Roll out the planar backend and label the outcome."""
    plan = plan_throw(policy, seed)
    if plan.launch is None:
        metadata = {"source": "planar_sim", "abort_reason": "unsupported_release_state"}
        return Trial(trial_id, session_id, plan.swing_state, plan.swing_id,
                     None, None, Outcome("aborted"), metadata)
    launch = plan.launch
    outcome, metadata = _fly(launch.z_m, launch.vz_m_s, launch.vx_m_s, launch.pitch_rad,
                             launch.pitch_rate_rad_s, launch.inertia_growth)
    metadata.update(source="planar_sim", decision_to_detach_s=launch.delay_s)
    return Trial(trial_id, session_id, plan.swing_state, plan.swing_id,
                 plan.release_state, plan.release_id, outcome, metadata)


def collect_trials(sessions, seed):
    rng = random.Random(seed)
    for session in range(sessions):
        pairs = [(swing, release) for swing in SWING_IDS for release in RELEASE_IDS]
        rng.shuffle(pairs)
        for attempt, (swing, release) in enumerate(pairs):
            session_id = f"sim-{session + 1:02}"
            yield simulate_trial(
                FixedPolicy(swing, release), rng.getrandbits(64),
                f"{session_id}-{attempt + 1:02}", session_id,
            )
