import io
import json
import unittest
from dataclasses import replace

from bottle_flip.baseline import fit_baseline
from bottle_flip.data import Outcome, SwingState, Trial, split_sessions, write_trials
from bottle_flip.iql import TwoStageIQL, UnsupportedStateError, fit_iql
from bottle_flip.isaac_demo import GraspPlayback
from bottle_flip.reward import score
from bottle_flip.simulator import collect_trials, plan_throw, simulate_trial


class BottleFlipTest(unittest.TestCase):
    def test_revo2_preview_opens_and_matches_launch_pose(self):
        class Hand:
            def pose(self, center, pitch, curl, offset):
                self.curl = curl

        trial = next(collect_trials(1, seed=9))
        plan = plan_throw(fit_baseline([trial]), seed=3)
        playback = GraspPlayback(0.5, plan)
        center, pitch = playback.bottle_pose(playback.detach_at)
        self.assertAlmostEqual(center[0], 0.5)
        self.assertAlmostEqual(center[2], plan.launch.z_m)
        self.assertAlmostEqual(pitch, plan.launch.pitch_rad)
        hand = Hand()
        for time, curl in ((0.0, 0.0), (0.72, 1.0), (playback.detach_at + 0.4, 0.0)):
            playback.move_hand(hand, time)
            self.assertEqual(hand.curl, curl)

    def test_rewards_distinguish_standing_from_near_misses(self):
        landing = Outcome("landed", -360, 3, True, True, 2)
        self.assertTrue(score(landing).success)
        self.assertEqual(score(landing).reward, 1.0)
        self.assertFalse(score(replace(landing, bottom_first=False)).success)
        self.assertFalse(score(replace(landing, flip_deg=720)).success)
        fallen = score(replace(landing, upright_s=1.99))
        self.assertFalse(fallen.success)
        self.assertLessEqual(fallen.reward, 0.15)
        self.assertEqual(score(replace(landing, in_target=False)).reward, 0.0)
        self.assertIsNone(score(Outcome("unobserved")))

    def test_trial_roundtrip_preserves_actions_and_supports_early_abort(self):
        trial = next(collect_trials(1, seed=17))
        stream = io.StringIO()
        write_trials(stream, [trial])
        self.assertEqual(Trial.from_dict(json.loads(stream.getvalue())), trial)
        landing = Outcome("landed", 360, 3, True, True, 2)
        trial = replace(trial, release_id=2, outcome=landing)
        aborted = replace(trial, trial_id="aborted", release_state=None,
                          release_id=None, outcome=Outcome("aborted"))
        self.assertEqual(Trial.from_dict(aborted.to_dict()), aborted)
        missed = replace(trial, trial_id="missed", release_id=0, outcome=Outcome("no_release"))
        trained = fit_iql([trial, aborted, missed], expectile=0.75)
        self.assertEqual(len(trained.swing.support_actions), 3)
        self.assertEqual(len(trained.release.support_actions), 2)
        # Terminal Q values are 0 and 1, so the expectile at s1 is 0.75.
        # The first-stage Q then averages two continuations and one terminal zero.
        self.assertAlmostEqual(trained.swing.critic[0, trial.swing_id], 0.5)
        with self.assertRaisesRegex(ValueError, "both"):
            Trial.from_dict({**trial.to_dict(), "release_id": None})

    def test_fixed_policy_counts_aborts_and_skips_unobserved_trials(self):
        trial = next(collect_trials(1, seed=4))
        success = Outcome("landed", 360, 3, True, True, 2)
        first = replace(trial, swing_id=0, release_id=1, outcome=success)
        aborted = replace(trial, trial_id="aborted", swing_id=0, release_state=None,
                          release_id=None, outcome=Outcome("aborted"))
        second = replace(trial, trial_id="second", swing_id=4, release_id=1, outcome=success)
        missing = replace(second, trial_id="missing", outcome=Outcome("unobserved"))
        self.assertEqual(fit_baseline([first, aborted, second, missing]).swing_id, 4)

    def test_session_split_and_loaded_policy(self):
        partitions = split_sessions(list(collect_trials(5, seed=7)), seed=7)
        for rows in partitions.values():
            for other in partitions.values():
                if rows is not other:
                    self.assertFalse({trial.session_id for trial in rows}
                                     & {trial.session_id for trial in other})
        fitted = fit_iql(partitions["train"])
        restored = TwoStageIQL.from_dict(json.loads(json.dumps(fitted.to_dict())))
        original_trial = simulate_trial(fitted, 13, "check", "fresh")
        self.assertEqual(simulate_trial(restored, 13, "check", "fresh"), original_trial)
        with self.assertRaises(UnsupportedStateError):
            restored.choose_swing(SwingState(10.0, 0.0))
        with self.assertRaises(UnsupportedStateError):
            restored.choose_swing(original_trial.swing_state, available=[])


if __name__ == "__main__":
    unittest.main()
