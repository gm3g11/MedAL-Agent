"""PAAL Accuracy Predictor (AP).

Faithful port of the ``Acc_Predictor`` in shijun18/PAAL-MedSeg
(model/predictor.py): ResNet-18 backbone, sigmoid head, per-class output.
Reference: Yi et al., "Predictive Accuracy-Based Active Learning for
Medical Image Segmentation" (IJCAI 2024).

AP input:  concat(image, softmax(seg_logits))  →  (C_img + C_seg, H, W)
AP output: predicted per-class accuracy/Dice in [0, 1]   →  (C_seg,)
Loss:      MSE against actual per-class HARD Dice computed from
           argmax(seg_logits) vs argmax(target onehot).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- ResNet-18 building blocks (verbatim from PAAL-MedSeg/model/predictor.py)

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class AccuracyPredictor(nn.Module):
    """ResNet-18 → sigmoid → per-class predicted Dice in [0, 1]. Matches the
    official PAAL-MedSeg Acc_Predictor (BasicBlock, layers=[2,2,2,2])."""

    def __init__(self, image_channels: int, num_classes: int, final_drop: float = 0.5):
        super().__init__()
        in_ch = image_channels + num_classes
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(final_drop) if final_drop > 0 else None
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        if self.drop is not None:
            x = self.drop(x)
        return torch.sigmoid(self.fc(x))


@torch.no_grad()
def hard_dice_per_class(probs: torch.Tensor, mask: torch.Tensor,
                        num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    """Hard argmax-Dice from softmax probs and integer masks.

    Matches the official PAAL-MedSeg `compute_dice(reduction='none')`:
    take argmax(probs) per pixel, then binary Dice per class.
    Returns (B, C) per-class hard Dice in [0, 1].

    Per-sample convention (matches the official `reduction='none'`):
    if a class is entirely absent from BOTH pred and gt for a sample,
    set Dice=1.0 (no error, trivially correct)."""
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    pred = torch.argmax(probs, dim=1)             # (B, H, W)
    B = probs.shape[0]
    out = torch.ones(B, num_classes, device=probs.device)
    for c in range(num_classes):
        p_c = (pred == c).float()
        g_c = (mask == c).float()
        p_sum = p_c.sum(dim=(1, 2))
        g_sum = g_c.sum(dim=(1, 2))
        inter = (p_c * g_c).sum(dim=(1, 2))
        denom = p_sum + g_sum
        # if both absent for a sample, leave as 1.0; else compute Dice
        present = (denom > 0)
        dice = (2.0 * inter + eps) / (denom + eps)
        out[:, c] = torch.where(present, dice, torch.ones_like(dice))
    return out.clamp(0.0, 1.0)
