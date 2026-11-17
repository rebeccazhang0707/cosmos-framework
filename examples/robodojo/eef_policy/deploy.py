"""Simulator-side native IK; the model server never emits joint actions."""

import json
import os
from pathlib import Path

from cosmos_framework.inference.robodojo_eef_control import (
    add_live_poses,
    native_joint_action,
    observation_pose,
    validate_packet,
)


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    log = Path(TASK_ENV.save_dir) / "eef_native_ik.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    chunk = 0
    with log.open("a", buffering=1) as stream:
        while not TASK_ENV.is_episode_end():
            obs = add_live_poses(TASK_ENV, TASK_ENV.get_obs())
            if chunk == 0:
                _, audit = native_joint_action(TASK_ENV.robot_manager, observation_pose(obs))
                stream.write(json.dumps({"audit": "measured_pose_native_ik", "arms": audit}) + "\n")
                if not all(arm["ik_success"] for arm in audit.values()):
                    raise RuntimeError("Native IK failed on the measured starting EEF pose")
            model_client.call(func_name="update_obs", obs=obs)
            targets = validate_packet(model_client.call(func_name="get_action"))
            for index, target in enumerate(targets):
                command, diagnostics = native_joint_action(TASK_ENV.robot_manager, target)
                stream.write(json.dumps({"chunk": chunk, "action_index": index, "arms": diagnostics}) + "\n")
                TASK_ENV.take_action(command)
                # Short end-to-end smoke runs must never be counted as task SR.
                limit = int(os.environ.get("ROBODOJO_EEF_SMOKE_STEPS", "0"))
                if limit and chunk * 16 + index + 1 >= limit:
                    return
                if TASK_ENV.is_episode_end():
                    break
                if index + 1 < len(targets):
                    TASK_ENV.get_obs()  # preserve benchmark's video/observation cadence
            chunk += 1


def eval_one_episode_batch(TASK_ENV, model_client):
    raise NotImplementedError("EEF adapter requires eval_batch=false")
