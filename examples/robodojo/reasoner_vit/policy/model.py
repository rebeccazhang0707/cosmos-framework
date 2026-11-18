"""Keep the verified joint adapter and require the image-conditioned model."""

from XPolicyLab.policy.Cosmos3_Edge_RoboDojo.model import Model as JointModel


class Model(JointModel):
    def __init__(self, model_cfg):
        super().__init__(model_cfg)
        from cosmos_framework.model.generator.omni_mot_reasoner_vision_model import OmniMoTReasonerVisionModel

        if not isinstance(self.service.model, OmniMoTReasonerVisionModel):
            raise TypeError("ViT evaluation must load the image-conditioned reasoner WAM model")
