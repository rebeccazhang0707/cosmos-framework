"""Edge480 WAM: current state plus16 future absolute joint targets."""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.callbacks.robodojo import RoboDojoTrainingMonitor
from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.robodojo_policy_dataset import get_robodojo_policy_sft_dataset
from cosmos_framework.data.generator.processors import build_processor
from cosmos_framework.utils.lazy_config import LazyCall as L

action_policy_robodojo_all_tasks_wam = copy.deepcopy(action_policy_droid_nano)
c = action_policy_robodojo_all_tasks_wam
c["model"]["config"] = copy.deepcopy(EDGE_MODEL_CONFIG)
m = c["model"]["config"]
m["resolution"] = "480"
m["vlm_config"]["tokenizer"] = L(build_processor)(
    tokenizer_type="${oc.env:ROBODOJO_TOKENIZER_PATH,nvidia/Cosmos3-Edge}"
)
m["vlm_config"]["pretrained_weights"]["enabled"] = False
m["tokenizer"]["encode_exact_durations"] = [17]
m["max_num_tokens_after_packing"] = -1
m["compile"]["enabled"] = False
c["checkpoint"].update(load_training_state=False)
c["trainer"].update(seed=42, save_zero_checkpoint=True)
c["trainer"]["callbacks"]["skip_nan_step"]["max_consecutive_nan"] = 1
c["trainer"]["callbacks"]["robodojo_monitor"] = L(RoboDojoTrainingMonitor)(canvas_width=736, canvas_height=544)
d = c["dataloader_train"]
d["dataset_name"] = "robodojo"
d["dataloader"].update(batch_size=1, num_workers=8, prefetch_factor=1, in_order=True)
d["dataloader"]["datasets"] = dict(
    robodojo=dict(
        ratio=1,
        dataset=L(get_robodojo_policy_sft_dataset)(
            prepared_root="${oc.env:ROBODOJO_PREPARED_ROOT}",
            tokenizer_config="${model.config.vlm_config.tokenizer}",
            max_action_dim="${model.config.max_action_dim}",
            chunk_length=16,
            mode="wam",
            resolution="480",
            canvas_width=736,
            canvas_height=544,
        ),
    )
)

ConfigStore.instance().store(
    group="experiment", package="_global_", name="action_policy_robodojo_all_tasks_wam", node=c
)
