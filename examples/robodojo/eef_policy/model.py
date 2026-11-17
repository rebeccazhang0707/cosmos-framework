"""Evaluation-only adapter for the normalized chunk-relative EEF20 RoboDojo checkpoint."""

import hashlib
import json
from pathlib import Path

import yaml
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_batch_size,
    get_robot_action_dim_info,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.env_cfg_type != "arx_x5" or model_cfg["action_type"] != "ee":
            raise ValueError("Checkpoint requires arx_x5 normalized chunk-relative EEF20 control")
        self.dims = get_robot_action_dim_info(self.env_cfg_type)
        if sum(self.dims["arm_dim"]) + sum(self.dims["ee_dim"]) != 14:
            raise ValueError("Native X5 IK requires two six-joint arms and two grippers")
        self.batch_size = get_batch_size(self.env_cfg_type)
        config = yaml.safe_load(Path(model_cfg["config_file"]).read_text())
        dataset = config["dataloader_train"]["dataloader"]["datasets"]["robodojo"]["dataset"]
        prepared = Path(dataset["prepared_root"])
        manifest = json.loads((prepared / "manifest.json").read_text())
        if (
            manifest["action_representation"],
            manifest["action_dim"],
            manifest["domain_id"],
            dataset["chunk_length"],
        ) != ("eef_relative_chunk", 20, 17, 16):
            raise ValueError("Checkpoint must use normalized relative EEF20, domain17, chunk16")
        normalization = Path(model_cfg["normalization_file"]).read_bytes()
        if normalization != (prepared / "normalization.json").read_bytes():
            raise ValueError("Inference normalizers differ from checkpoint's training normalizers")
        self.stats = json.loads(normalization)
        print(json.dumps({"eef_normalization_sha256": hashlib.sha256(normalization).hexdigest()}), flush=True)
        policy_dir = Path(__file__).resolve().parent
        checkpoint = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs

        class RobotServerArgs(ActionServerArgs):
            def build_setup_overrides(self):
                setup = super().build_setup_overrides()
                # This adapter consumes fixed benchmark prompts and returns
                # EEF targets, without decoding generated media.
                setup.guardrails = False
                return setup

        self.service = ActionModelService(
            RobotServerArgs(
                checkpoint=CheckpointOverrides(checkpoint_path=str(checkpoint), config_file=model_cfg["config_file"]),
                output_dir=Path(model_cfg["output_dir"]) / f"port_{model_cfg.get('port', 0)}",
                seed=int(model_cfg.get("seed") or 0),
                guidance=1.0,
                num_steps=int(model_cfg.get("num_steps", 30)),
                fps=25,
                action_chunk_size=16,
                max_action_dim=64,
                raw_action_dim=20,
                format_prompt_as_json=True,
            )
        )
        self.reset()

    def reset(self):
        self.observations = {}
        self.single_observation = None

    def update_obs(self, obs):
        self.single_observation = obs

    def update_obs_batch(self, obs_list):
        for index, observation in enumerate(obs_list):
            self.observations[int(observation.get("env_idx", index))] = observation

    def _predict(self, observations):
        from cosmos_framework.inference.robodojo_eef_control import decode_chunk, observation_pose, state20
        from cosmos_framework.inference.robodojo_eef_policy import predict_normalized_actions

        anchors = [observation_pose(obs) for obs in observations]
        states = [state20(anchor, self.stats) for anchor in anchors]
        chunks = predict_normalized_actions(self.service, observations, states)
        return [decode_chunk(anchor, chunk, self.stats) for anchor, chunk in zip(anchors, chunks, strict=True)]

    def get_action(self):
        if self.single_observation is None:
            raise ValueError("No observation has been supplied")
        return self._predict([self.single_observation])[0]

    def get_action_batch(self, env_idx_list=None):
        raise NotImplementedError("Only verified single-environment EEF rollouts are supported")
