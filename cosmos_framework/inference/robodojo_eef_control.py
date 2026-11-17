"""EEF20 transport and native RoboDojo IK boundary (NumPy-only, safe in Isaac's env).

Training uses separate state/action normalizers. All sixteen relative targets
share the current chunk's anchor. RoboDojo world poses are environment-local
xyz + wxyz; its native RobotManager handles robot-base/gripper transforms.
"""

import numpy as np

from cosmos_framework.data.generator.action.utils.robodojo_eef import absolute20, denormalize, normalize, recover16

REPRESENTATION = "robodojo_env_world_pose16_wxyz_v1"


def finite(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite {shape}, got {array.shape}")
    return array


def observation_pose(observation):
    state = observation["state"]
    parts = []
    for arm in ("left", "right"):
        parts.extend([finite(state[f"{arm}_ee_pose"], (7,)), finite(state[f"{arm}_ee_joint_state"], (1,))])
    return np.concatenate(parts)


def state20(anchor, stats):
    return normalize(absolute20(finite(anchor, (16,))), stats["state"])


def decode_chunk(anchor, normalized_actions, stats):
    actions = finite(normalized_actions, (16, 20))
    targets = recover16(finite(anchor, (16,)), denormalize(actions, stats["action"]))
    finite(targets, (16, 16))
    # Grip is absolute in the labels; only the physical actuator boundary clips.
    return {"representation": REPRESENTATION, "targets": targets.tolist()}


def validate_packet(packet):
    if not isinstance(packet, dict) or packet.get("representation") != REPRESENTATION:
        raise ValueError("EEF rollout requires a typed world-pose packet; joint/raw20 actions are forbidden")
    return finite(packet["targets"], (16, 16))


def native_joint_action(manager, target16, env_idx=0):
    """Call native cuRobo IK per arm, preserving native joint ordering/control.

    Failed IK holds that arm at its measured joint position; gripper targets
    still apply, as in the benchmark's native EE controller. Branch jumps are
    reported without silently changing the official IK solution or controller.
    """
    target = finite(target16, (16,)).reshape(2, 8)
    command, diagnostics = {}, {}
    for index, arm in enumerate(("left", "right")):
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        current = finite(manager.get_joint(robot, env_idx_list=[env_idx])[env_idx], (6,))
        pose = target[index, :7]
        if not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-5):
            raise ValueError("Native IK requires unit wxyz quaternion")
        result = manager.solve_ik(target_pose=pose.tolist(), env_idx=env_idx, robot=robot, trans="world")
        success = result["status"] == "Success"
        joint = finite(result["joint_value"], (6,)) if success else current
        command[manager.process_name(robot.arm_name)] = joint.tolist()
        grip = float(np.clip(target[index, 7], 0, 1))
        command[manager.process_name(robot.gripper_name)] = [grip]
        diagnostics[arm] = {
            "ik_success": success,
            "max_joint_delta_rad": float(np.abs(joint - current).max()),
            "gripper_clipped": bool(grip != target[index, 7]),
            "eef_world_target": target[index].tolist(),
            "joint_target": joint.tolist(),
        }
    return command, diagnostics


def add_live_poses(env, observation):
    """Always use measured simulator poses, regardless of observation YAML flags."""
    for arm in ("left", "right"):
        robot = env.robot_manager.get_robot_by_arm_name(f"{arm}_arm")
        observation["state"][f"{arm}_ee_pose"] = env.robot_manager.get_real_endpose(robot=robot, env_idx_list=[0])[0]
    observation_pose(observation)  # fail before inference on missing/invalid fields
    return observation
