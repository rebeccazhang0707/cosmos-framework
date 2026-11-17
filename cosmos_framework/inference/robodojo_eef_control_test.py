"""Control-boundary checks independent of CUDA/Isaac/model dependencies."""

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action.utils.robodojo_eef import absolute20, normalize, pose_errors, relative20
from cosmos_framework.inference.robodojo_eef_control import (
    decode_chunk,
    native_joint_action,
    observation_pose,
    state20,
    validate_packet,
)


class NativeManager:
    def __init__(self):
        self.calls = []

    def get_robot_by_arm_name(self, name):
        return SimpleNamespace(arm_name=name, gripper_name=name.replace("arm", "ee"))

    def process_name(self, name):
        return name + "_joint_state"

    def get_joint(self, robot, env_idx_list):
        return {0: np.arange(6) / 10}

    def solve_ik(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["robot"].arm_name == "right_arm":
            return {"status": "Fail"}
        return {"status": "Success", "joint_value": np.arange(6) / 10 + 0.01}


class EEFControlTests(unittest.TestCase):
    def test_nonidentity_anchor_separate_normalizers_and_no_cumulative_decode(self):
        rng = np.random.default_rng(27)
        poses = rng.normal(size=(17, 2, 8))
        poses[..., 3:7] = Rotation.random(34, random_state=rng).as_quat()[:, [3, 0, 1, 2]].reshape(17, 2, 4)
        poses = poses.reshape(17, 16)
        stats = {
            key: dict(center=rng.normal(size=20), scale=rng.uniform(0.1, 2, size=20)) for key in ("state", "action")
        }
        encoded = normalize(relative20(poses[0], poses[1:]), stats["action"])
        packet = decode_chunk(poses[0], encoded, stats)
        error_p, error_r = pose_errors(validate_packet(packet), poses[1:])
        self.assertLess(error_p.max(), 1e-6)
        self.assertLess(error_r.max(), 1e-6)
        np.testing.assert_allclose(state20(poses[0], stats), normalize(absolute20(poses[0]), stats["state"]))

    def test_native_ik_world_mode_failure_holds_joints_and_clips_absolute_grip(self):
        manager = NativeManager()
        pose = np.array([0.1, 0.2, 1, 1, 0, 0, 0, 2] * 2)
        pose[-1] = -1
        command, diag = native_joint_action(manager, pose)
        self.assertEqual(len(manager.calls), 2)
        self.assertTrue(all(c["trans"] == "world" for c in manager.calls))
        np.testing.assert_allclose(manager.calls[0]["target_pose"], pose[:7])
        np.testing.assert_allclose(command["left_arm_joint_state"], np.arange(6) / 10 + 0.01)
        np.testing.assert_allclose(command["right_arm_joint_state"], np.arange(6) / 10)
        self.assertEqual(command["left_ee_joint_state"], [1])
        self.assertEqual(command["right_ee_joint_state"], [0])
        self.assertFalse(diag["right"]["ik_success"])
        self.assertTrue(diag["left"]["gripper_clipped"])

    def test_reject_joint_actions_raw20_nonfinite_and_bad_quaternion(self):
        for packet in ([0] * 14, np.zeros((16, 20)), {"targets": np.zeros((16, 16))}):
            with self.assertRaises(ValueError):
                validate_packet(packet)
        with self.assertRaises(ValueError):
            native_joint_action(NativeManager(), np.zeros(16))
        with self.assertRaises(ValueError):
            native_joint_action(NativeManager(), np.full(16, np.nan))
        with self.assertRaises(KeyError):
            observation_pose({"state": {"left_arm_joint_state": np.zeros(6)}})


if __name__ == "__main__":
    unittest.main()
