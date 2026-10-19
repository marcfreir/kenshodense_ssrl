"""
deeplabv3.py — DeepLabV3 with a ResNet-50 (1x / 2x / 4x width) backbone.

Modified version of the DeepLabV3 implementation from the Minerva project:
https://github.com/discovery-unicamp/Minerva/blob/main/minerva/models/nets/image/deeplabv3.py
We thank all the authors of the Minerva project.

Changes with respect to the original:
  * ``WideResNet50`` / ``build_resnet50``: width-multiplied ResNet-50
    (SimCLR / BYOL "RN50 (2x)", "RN50 (4x)").
  * ``DeepLabV3Backbone.out_channels`` (= 2048 * width_multiplier), so the
    KenshoDense heads and the DeepLabV3 head size themselves automatically.
"""

from typing import Any, Dict, Optional, Sequence, Tuple
import os
from collections import OrderedDict
from torch import Tensor, nn, optim
from torchmetrics import Metric
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet, Bottleneck, resnet50
from torchvision.models.segmentation.deeplabv3 import ASPP
import torch
from minerva.models.nets.base import SimpleSupervisedModel


# ---------------------------------------------------------------------------
# Width-multiplied ResNet-50 (SimCLR / BYOL "RN50 (2x)", "RN50 (4x)")
# ---------------------------------------------------------------------------

SUPPORTED_WIDTH_MULTIPLIERS = (1, 2, 4)


class WideResNet50(ResNet):
    """ResNet-50 whose channel count is multiplied by ``width_multiplier`` in
    EVERY layer (stem, bottleneck inner convs and bottleneck outputs), exactly
    as in SimCLR / BYOL ("ResNet-50 (2x)", "ResNet-50 (4x)").

    Depth (3-4-6-3 bottleneck blocks) is unchanged; only width changes.

        k=1 -> stem 64,  stage outputs 256 / 512 / 1024 / 2048   (~24M params)
        k=2 -> stem 128, stage outputs 512 / 1024 / 2048 / 4096  (~94M params)
        k=4 -> stem 256, stage outputs 1024 / 2048 / 4096 / 8192 (~375M params)

    NOTE: this is NOT torchvision's ``wide_resnet50_2``. That model only widens
    the inner 3x3 conv of each bottleneck (width_per_group=128); its output is
    still 2048 channels and it has ~69M params, so it is not the SimCLR/BYOL 2x.

    Module names (conv1, bn1, layer1..layer4, fc) are identical to torchvision's
    resnet50, so the state-dict layout is the same apart from tensor shapes.
    """

    def __init__(
        self,
        width_multiplier: int = 1,
        replace_stride_with_dilation: Optional[Sequence[bool]] = None,
        layers: Sequence[int] = (3, 4, 6, 3),
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ):
        # Bypass ResNet.__init__ (its stem/stage widths are hard-coded) but
        # reuse its _make_layer, which reads the attributes set below.
        nn.Module.__init__(self)
        k = int(width_multiplier)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self._norm_layer = nn.BatchNorm2d
        self.width_multiplier = k
        self.inplanes = 64 * k
        self.dilation = 1
        self.groups = 1
        self.base_width = 64  # bottleneck inner width = planes, which we scale by k

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64 * k, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128 * k, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck, 256 * k, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Bottleneck, 512 * k, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * k * Bottleneck.expansion, num_classes)

        # Same initialisation as torchvision's ResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)


def build_resnet50(
    width_multiplier: int = 1,
    replace_stride_with_dilation: Optional[Sequence[bool]] = None,
) -> ResNet:
    """1x -> torchvision's resnet50 (bit-identical to the previous behaviour,
    so existing 1x runs/seeds are unaffected); 2x/4x -> WideResNet50."""
    if width_multiplier not in SUPPORTED_WIDTH_MULTIPLIERS:
        raise ValueError(
            f"width_multiplier must be one of {SUPPORTED_WIDTH_MULTIPLIERS}, "
            f"got {width_multiplier}."
        )
    if width_multiplier == 1:
        return resnet50(replace_stride_with_dilation=replace_stride_with_dilation)
    return WideResNet50(width_multiplier, replace_stride_with_dilation=replace_stride_with_dilation)


class DeepLabV3(SimpleSupervisedModel):
    """A DeeplabV3 with a ResNet50 (1x / 2x / 4x width) backbone

    References
    ----------
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.
    "Rethinking Atrous Convolution for Semantic Image Segmentation", 2017
    """

    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        pred_head: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
        learning_rate: float = 0.001,
        num_classes: int = 6,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        train_metrics: Optional[Dict[str, Metric]] = None,
        val_metrics: Optional[Dict[str, Metric]] = None,
        test_metrics: Optional[Dict[str, Metric]] = None,
        optimizer: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        width_multiplier: int = 1,
    ):
        """
        Initializes a DeepLabV3 model.

        Parameters
        ----------
        backbone: Optional[nn.Module]
            The backbone network. Defaults to None, which will use a ResNet50
            backbone of the given ``width_multiplier``.
        pred_head: Optional[nn.Module]
            The prediction head network. Defaults to None, which will use a
            DeepLabV3PredictionHead whose input channels match the backbone's
            output channels (2048 * width_multiplier).
        loss_fn: Optional[nn.Module]
            The loss function. Defaults to None, which will use a
            CrossEntropyLoss.
        learning_rate: float
            The learning rate for the optimizer. Defaults to 0.001.
        num_classes: int
            The number of classes for prediction. Defaults to 6.
        pretrained: bool
            Whether to use pretrained weights. Defaults to False.
        weights_path: Optional[str]
            Path to local pretrained weights file. Defaults to None.
        train_metrics, val_metrics, test_metrics: Optional[Dict[str, Metric]]
            Metrics computed during each stage. Default None.
        optimizer: type
            Optimizer class. Defaults to `torch.optim.Adam`.
        optimizer_kwargs : dict, optional
            Additional kwargs passed to the optimizer constructor.
        lr_scheduler : type, optional
            Learning rate scheduler class. Defaults to None.
        lr_scheduler_kwargs : dict, optional
            Additional kwargs passed to the scheduler constructor.
        output_shape: Optional[Tuple[int, ...]]
            The output shape of the model. Defaults to the input shape.
        width_multiplier: int
            1, 2 or 4. Only used when ``backbone`` is None. When a backbone is
            passed, its own ``out_channels`` determines the head's input size.
        """
        backbone = backbone or DeepLabV3Backbone(
            num_classes=num_classes,
            pretrained=pretrained,
            weights_path=weights_path,
            width_multiplier=width_multiplier,
        )
        head_in_channels = getattr(backbone, "out_channels", 2048 * width_multiplier)
        pred_head = pred_head or DeepLabV3PredictionHead(
            in_channels=head_in_channels, num_classes=num_classes
        )
        loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.output_shape = output_shape

        super().__init__(
            backbone=backbone,
            fc=pred_head,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        input_shape = self.output_shape or x.shape[-2:]
        h = self.backbone(x)
        if isinstance(h, OrderedDict):
            h = h["out"]
        z = self.fc(h)
        return nn.functional.interpolate(
            z, size=input_shape, mode="bilinear", align_corners=False
        )

    def _loss_func(self, y_hat: Tensor, y: Tensor) -> Tensor:
        return self.loss_fn(y_hat, y.squeeze(1).long())


class DeepLabV3Backbone(nn.Module):
    """A ResNet50 (1x / 2x / 4x width) backbone for DeepLabV3.

    The attribute ``out_channels`` (= 2048 * width_multiplier) lets downstream
    code (KenshoDense heads, DeepLabV3 head) size itself automatically.
    """

    def __init__(
        self,
        num_classes: int = 6,
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        width_multiplier: int = 1,
    ):
        """
        Parameters
        ----------
        num_classes: int
            Kept for API compatibility. Default 6.
        pretrained: bool
            Whether to use pretrained weights. ImageNet weights from
            torchvision only exist for width_multiplier=1; for 2x/4x you must
            supply ``weights_path``.
        weights_path: Optional[str]
            Path to a local pretrained weights file.
        width_multiplier: int
            1 (standard ResNet-50), 2 (RN50 2x) or 4 (RN50 4x).
        """
        super().__init__()
        self.width_multiplier = int(width_multiplier)
        self.out_channels = 2048 * self.width_multiplier
        dilation = [False, True, True]

        if pretrained and weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            RN50model = build_resnet50(self.width_multiplier, dilation)

            state_dict = torch.load(weights_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith("fc.")
            }
            missing_keys, unexpected_keys = RN50model.load_state_dict(
                filtered_state_dict, strict=False
            )
            if missing_keys:
                print(f"Warning: Missing keys when loading pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected_keys}")
            print(f"Successfully loaded pretrained weights from {weights_path}")

        elif pretrained and weights_path is None:
            if self.width_multiplier != 1:
                raise ValueError(
                    "torchvision only ships ImageNet weights for the 1x ResNet-50. "
                    "For width_multiplier=2/4 pass weights_path, or use pretrained=False."
                )
            RN50model = resnet50(
                weights=ResNet50_Weights.IMAGENET1K_V1,
                replace_stride_with_dilation=dilation,
            )
            print("Successfully loaded ImageNet pretrained weights from torchvision")
        else:
            RN50model = build_resnet50(self.width_multiplier, dilation)

        self.RN50model = RN50model

    def freeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.RN50model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """(N, 3, H, W) -> (N, 2048 * width_multiplier, H/8, W/8)."""
        x = self.RN50model.conv1(x)
        x = self.RN50model.bn1(x)
        x = self.RN50model.relu(x)
        x = self.RN50model.maxpool(x)
        x = self.RN50model.layer1(x)
        x = self.RN50model.layer2(x)
        x = self.RN50model.layer3(x)
        x = self.RN50model.layer4(x)
        return x


class DeepLabV3PredictionHead(nn.Sequential):
    """The prediction head for DeepLabV3"""

    def __init__(
        self,
        in_channels: int = 2048,
        num_classes: int = 6,
        atrous_rates: Sequence[int] = (12, 24, 36),
    ) -> None:
        super().__init__(
            ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, input) -> Tensor:
        assert input.shape[0] > 1, "Batch size must be greater than 1 due to BatchNorm"
        return super().forward(input)


