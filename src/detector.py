import collections

import torch

import fpn
from third_party import resnet
from third_party.detectron2 import postprocess
from third_party.detectron2 import regression
from third_party.detectron2 import anchors
from third_party.detectron2 import retinanet_head


class Detector(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        confidence: float = 0.05,
        num_detections_per_image: int = 100
    ) -> None:
        super().__init__()
        self.num_detections_per_image = num_detections_per_image
        self.confidence = confidence
        self.fpn_channels = 128
        self.fpn_levels = [4, 5, 6, 7, 8]
        self.backbone = resnet.resnet18()
        self.backbone.delete_classification_head()

        self.fpn = self._load_fpn("retinanet", self.backbone.get_pyramid_channels())
        self.anchors = anchors.AnchorGenerator(
            img_height=512,
            img_width=512,
            pyramid_levels=self.fpn_levels,
            aspect_ratios=[0.5, 1.0, 2.0],
            sizes=[16, 32, 64, 128, 256],
            anchor_scales=[1.0, 1.25, 1.5],
        )

        # Create the retinanet head.
        self.retinanet_head = retinanet_head.RetinaNetHead(
            num_classes,
            in_channels=self.fpn_channels,
            anchors_per_cell=self.anchors.num_anchors_per_cell,
            num_convolutions=3,
            use_dw=False,
        )

        self.image_size = [512, 512, 512, 512]
        self.postprocess = postprocess.PostProcessor(
            num_classes=num_classes,
            image_size=self.image_size,
            all_anchors=self.anchors.all_anchors,
            regressor=regression.Regressor(),
            max_detections_per_image=num_detections_per_image,
            score_threshold=confidence,
            nms_threshold=0.2,
        )

        self.eval()


    def _load_fpn(self, fpn_name, features) -> torch.nn.Module:
        if "retinanet" in fpn_name:
            fpn_ = fpn.FPN(
                in_channels=features[-3:],
                out_channels=self.fpn_channels,
                num_levels=len(self.fpn_levels),
                use_dw=False,
            )

        return fpn_

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        levels = self.backbone.forward_pyramids(x)
        # Only keep the levels specified during construction.
        levels = collections.OrderedDict(
            [item for item in levels.items() if item[0] in self.fpn_levels]
        )
        levels = self.fpn(levels)
        classifications, regressions = self.retinanet_head(levels)

        if self.training:
            return classifications, regressions
        else:
            return self.postprocess(classifications, regressions)
