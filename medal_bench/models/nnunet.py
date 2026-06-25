"""nnU-Net v2 wrapper with optional dropout.

We use nnU-Net v2 as a *library*: its ``dynamic_network_architectures``
PlainConvUNet builder gives us a clean 2D U-Net. We pass a non-None
``dropout_op`` so the network has dropout layers built in, which lets us:

  - run normal inference with ``net.eval()`` (dropout off, deterministic)
  - run MC dropout for P2 BALD by enabling dropout-only modules at eval time

The same architecture is used for ALL policies in BALD-included experiments.
P2 toggles MC-dropout mode at inference (K stochastic passes); all other
policies use plain eval mode.

build_unet_2d(...) is the only public entry point. enable_mc_dropout(net)
turns dropout back on for inference without touching BN running stats.
"""
from __future__ import annotations
from typing import Optional, Sequence

import torch
import torch.nn as nn


def build_unet_2d(
    input_channels: int = 1,
    num_classes: int = 2,
    features_per_stage: Sequence[int] = (32, 64, 128, 256, 512),
    n_conv_per_stage: int = 2,
    n_conv_per_stage_decoder: int = 2,
    kernel_sizes: Sequence[int] = (3, 3),
    strides_first: Sequence[int] = (1, 1),
    dropout_p: float = 0.1,
) -> nn.Module:
    """Build a 2D nnU-Net (PlainConvUNet) with dropout enabled.

    All non-architectural defaults come from nnU-Net v2's standard 2d plan.
    dropout_p=0.0 disables dropout (returns the standard arch); dropout_p>0
    inserts Dropout2d layers, which we later toggle for P2 MC dropout.

    Returns a ``nn.Module`` whose forward takes (B, C, H, W) and returns
    (B, num_classes, H, W) logits.
    """
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from dynamic_network_architectures.building_blocks.helper import (
        convert_dim_to_conv_op, get_matching_instancenorm,
    )

    n_stages = len(features_per_stage)
    conv_op = convert_dim_to_conv_op(2)
    norm_op = get_matching_instancenorm(conv_op)

    # nnU-Net's standard 2D strides: 1 at stage 0, 2 thereafter
    strides = [list(strides_first)]
    for _ in range(1, n_stages):
        strides.append([2, 2])

    if dropout_p > 0:
        dropout_op = nn.Dropout2d
        dropout_op_kwargs = {"p": dropout_p}
    else:
        dropout_op = None
        dropout_op_kwargs = None

    net = PlainConvUNet(
        input_channels=input_channels,
        n_stages=n_stages,
        features_per_stage=tuple(features_per_stage),
        conv_op=conv_op,
        kernel_sizes=tuple(list(kernel_sizes) for _ in range(n_stages)),
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=dropout_op,
        dropout_op_kwargs=dropout_op_kwargs,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )
    return net


def enable_mc_dropout(net: nn.Module) -> int:
    """For MC dropout at inference: set only Dropout modules to train mode
    while keeping BN/IN in eval (so running stats are not perturbed).

    Returns the number of dropout modules toggled (sanity: should be > 0
    if the net was built with dropout_p > 0).
    """
    n = 0
    net.eval()
    for m in net.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
            n += 1
    return n


def has_dropout(net: nn.Module) -> bool:
    return any(m.__class__.__name__.startswith("Dropout") for m in net.modules())
