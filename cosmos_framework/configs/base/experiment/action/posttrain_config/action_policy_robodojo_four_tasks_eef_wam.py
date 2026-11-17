"""Exp1 Edge WAM control: normalized chunk-relative EEF20 instead of joint14.

Both recipes use 960x736, 16 GPUs, global batch2560, and the exp1 optimizer.
The EEF manifest supplies normalized absolute EEF state and relative targets.
The isolated four-task control keeps domain17 to pair the initial embedding.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_robodojo_four_tasks_wam import (
    action_policy_robodojo_four_tasks_wam,
)

action_policy_robodojo_four_tasks_eef_wam = copy.deepcopy(action_policy_robodojo_four_tasks_wam)

ConfigStore.instance().store(
    group="experiment",
    package="_global_",
    name="action_policy_robodojo_four_tasks_eef_wam",
    node=action_policy_robodojo_four_tasks_eef_wam,
)
