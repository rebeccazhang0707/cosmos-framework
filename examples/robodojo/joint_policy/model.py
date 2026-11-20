"""Evaluation-only adapter for the local Cosmos3-Edge RoboDojo checkpoint."""

from pathlib import Path

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_batch_size,
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.env_cfg_type != "arx_x5" or model_cfg["action_type"] != "joint":
            raise ValueError("Checkpoint supports arx_x5 absolute joint control")
        self.dims = get_robot_action_dim_info(self.env_cfg_type)
        if sum(self.dims["arm_dim"]) + sum(self.dims["ee_dim"]) != 14:
            raise ValueError("Robot dimensions disagree with trained 14D contract")
        self.batch_size = get_batch_size(self.env_cfg_type)
        policy_dir = Path(__file__).resolve().parent
        checkpoint = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints", policy_dir=policy_dir)
        from cosmos_framework.inference.common.args import CheckpointOverrides
        from cosmos_framework.scripts.action_policy_server_libero import ActionModelService, ActionServerArgs

        class RobotServerArgs(ActionServerArgs):
            def build_setup_overrides(self):
                setup = super().build_setup_overrides()
                # This adapter consumes fixed benchmark prompts and returns
                # joint targets, without decoding generated media.
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
                raw_action_dim=14,
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
        from cosmos_framework.inference.robodojo_policy import predict_actions

        states = [pack_robot_state(obs, "joint", self.dims) for obs in observations]
        chunks = predict_actions(self.service, observations, states)
        return [unpack_robot_state(chunk, "joint", self.dims) for chunk in chunks]

    def get_action(self):
        if self.single_observation is None:
            raise ValueError("No observation has been supplied")
        return self._predict([self.single_observation])[0]

    def get_action_batch(self, env_idx_list=None):
        indices = list(env_idx_list) if env_idx_list is not None else list(range(self.batch_size))
        return self._predict([self.observations[int(index)] for index in indices])
