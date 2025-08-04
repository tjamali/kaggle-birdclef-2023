from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchlibrosa.augmentation import SpecAugmentation

from configs import Config
from transforms import interpolate, pad_framewise_output

# =============================================================================
# Logger
# =============================================================================
logger = logging.getLogger(__name__)

# Maintain original pattern: config refers to class, not instance.
config = Config


# =============================================================================
# Padding utility for "same" convs
# =============================================================================
def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(
    x,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups: int = 1,
):
    """
    TensorFlow-style 'SAME' convolution: pads input so output spatial dims match input / stride.
    """
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(
        x,
        [
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
        ],
    )
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """Wrapper over Conv2d to emulate TensorFlow 'SAME' padding behavior."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


# =============================================================================
# Initialization helpers
# =============================================================================
def init_layer(layer):
    """
    Xavier uniform initialization for layer weights and zero bias.
    """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    """
    Initialize batch norm to identity (weight=1, bias=0).
    """
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    """
    Legacy weight initializer based on class name heuristics.
    """
    classname = model.__class__.__name__
    if "Conv2d" in classname:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        if hasattr(model, "bias") and model.bias is not None:
            model.bias.data.fill_(0)
    elif "BatchNorm" in classname:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif "GRU" in classname:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif "Linear" in classname:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


# =============================================================================
# Interpolation / padding utilities
# =============================================================================
def interpolate_time(x: torch.Tensor, ratio: int):
    """
    Upsample in the time domain by repeating frames (simple nearest-like expansion).
    """
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """
    Resize framewise output to match original frame count via bilinear interpolation.
    """
    output = F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)
    return output


# =============================================================================
# Attention blocks
# =============================================================================
class AttBlockV2(nn.Module):
    def __init__(self, in_features, out_features, activation="linear"):
        super().__init__()
        self.activation = activation
        self.att = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (batch_size, n_features, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        return x  # fallback


class MultiheadAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads, activation="linear"):
        super().__init__()
        self.num_heads = num_heads
        self.activation = activation

        self.att_list = nn.ModuleList()
        self.cla_list = nn.ModuleList()
        for _ in range(self.num_heads):
            self.att_list.append(nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=True))
            self.cla_list.append(nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=True))

        self.init_weights()

    def init_weights(self):
        for i in range(self.num_heads):
            init_layer(self.att_list[i])
            init_layer(self.cla_list[i])

    def forward(self, x):
        norm_att_list = []
        cla_list = []
        att_cla_list = []
        for i in range(self.num_heads):
            norm_att = torch.softmax(torch.tanh(self.att_list[i](x)), dim=-1)
            cla = self.nonlinear_transform(self.cla_list[i](x))
            att_cla = torch.sum(norm_att * cla, dim=2)

            norm_att_list.append(norm_att)
            cla_list.append(cla)
            att_cla_list.append(att_cla)

        return att_cla_list, norm_att_list, cla_list

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        return x


# =============================================================================
# Core models
# =============================================================================
class TimmSED(nn.Module):
    def __init__(self, base_model_name, pretrained=False, num_classes=24, in_channels=1):
        super().__init__()
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64 // 2,
            time_stripes_num=2,
            freq_drop_width=8 // 2,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(config.n_mels)

        base_model = timm.create_model(base_model_name, pretrained=pretrained, in_chans=in_channels)
        layers = list(base_model.children())[:-2]
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.num_features
        self.in_features = in_features

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block1 = AttBlockV2(in_features, num_classes, activation="sigmoid")
        self.att_block2 = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input_data):
        x = input_data
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and random.random() < 0.25:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        x = self.encoder(x)
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = F.relu_(x)
        x = x.transpose(1, 2)

        (clipwise_output1, norm_att1, segmentwise_output1) = self.att_block1(x)
        (clipwise_output2, norm_att2, segmentwise_output2) = self.att_block2(x)

        logit1 = torch.sum(norm_att1 * self.att_block1.cla(x), dim=2)
        logit2 = torch.sum(norm_att2 * self.att_block2.cla(x), dim=2)

        segmentwise_logit1 = self.att_block1.cla(x).transpose(1, 2)
        segmentwise_logit2 = self.att_block2.cla(x).transpose(1, 2)

        segmentwise_output1 = segmentwise_output1.transpose(1, 2)
        segmentwise_output2 = segmentwise_output2.transpose(1, 2)

        interpolate_ratio1 = frames_num // segmentwise_output1.size(1)
        interpolate_ratio2 = frames_num // segmentwise_output2.size(1)

        framewise_output1 = interpolate(segmentwise_output1, interpolate_ratio1)
        framewise_output1 = pad_framewise_output(framewise_output1, frames_num)
        framewise_logit1 = interpolate(segmentwise_logit1, interpolate_ratio1)
        framewise_logit1 = pad_framewise_output(framewise_logit1, frames_num)

        framewise_output2 = interpolate(segmentwise_output2, interpolate_ratio2)
        framewise_output2 = pad_framewise_output(framewise_output2, frames_num)
        framewise_logit2 = interpolate(segmentwise_logit2, interpolate_ratio2)
        framewise_logit2 = pad_framewise_output(framewise_logit2, frames_num)

        clipwise_output = (clipwise_output1 + clipwise_output2) / 2.0
        logit = (logit1 + logit2) / 2.0
        framewise_output = (framewise_output1 + framewise_output2) / 2.0
        framewise_logit = (framewise_logit1 + framewise_logit2) / 2.0

        return {
            "clipwise_output": clipwise_output,
            "logit": logit,
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
        }


def load_pretrained_sed_checkpoint(model, checkpoint):
    """
    Load a pretrained SED checkpoint, stripping distributed wrappers if needed.
    """
    checkpoint_data = torch.load(checkpoint)
    checkpoint_dict = {}
    for k, v in checkpoint_data.get("state_dict", {}).items():
        if "num_batches_tracked" in k:
            continue
        if k.startswith("module."):
            checkpoint_dict[k[7:]] = v
        else:
            checkpoint_dict[k] = v
    model.load_state_dict(checkpoint_dict)


def create_pretrained_sed_model(base_model_name, pretrained, num_classes=562, in_channels=3):
    """
    Instantiate a TimmSED model and optionally load pretrained weights.
    """
    sed_model = TimmSED(base_model_name, pretrained=False, num_classes=num_classes, in_channels=in_channels)
    if pretrained:
        load_pretrained_sed_checkpoint(sed_model, config.pretrained_checkpoint)
    return sed_model


class SED(nn.Module):
    def __init__(self, base_model_name, pretrained=False, num_classes=264, in_channels=3):
        super().__init__()

        pretrained_model = create_pretrained_sed_model(base_model_name, pretrained)
        self.spec_augmenter = pretrained_model.spec_augmenter
        self.bn0 = nn.BatchNorm2d(config.n_mels)
        self.encoder = pretrained_model.encoder

        in_features = pretrained_model.in_features
        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        self.att_block1 = AttBlockV2(in_features, num_classes, activation="sigmoid")
        self.att_block2 = AttBlockV2(in_features, num_classes, activation="sigmoid")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input_data):
        x = input_data
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and random.random() < 0.25:
            x = self.spec_augmenter(x)

        x = x.transpose(2, 3)
        x = self.encoder(x)
        x = torch.mean(x, dim=2)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = F.relu_(x)
        x = x.transpose(1, 2)

        (clipwise_output1, norm_att1, segmentwise_output1) = self.att_block1(x)
        (clipwise_output2, norm_att2, segmentwise_output2) = self.att_block2(x)

        logit1 = torch.sum(norm_att1 * self.att_block1.cla(x), dim=2)
        logit2 = torch.sum(norm_att2 * self.att_block2.cla(x), dim=2)

        segmentwise_logit1 = self.att_block1.cla(x).transpose(1, 2)
        segmentwise_logit2 = self.att_block2.cla(x).transpose(1, 2)

        segmentwise_output1 = segmentwise_output1.transpose(1, 2)
        segmentwise_output2 = segmentwise_output2.transpose(1, 2)

        interpolate_ratio1 = frames_num // segmentwise_output1.size(1)
        interpolate_ratio2 = frames_num // segmentwise_output2.size(1)

        framewise_output1 = interpolate(segmentwise_output1, interpolate_ratio1)
        framewise_output1 = pad_framewise_output(framewise_output1, frames_num)
        framewise_logit1 = interpolate(segmentwise_logit1, interpolate_ratio1)
        framewise_logit1 = pad_framewise_output(framewise_logit1, frames_num)

        framewise_output2 = interpolate(segmentwise_output2, interpolate_ratio2)
        framewise_output2 = pad_framewise_output(framewise_output2, frames_num)
        framewise_logit2 = interpolate(segmentwise_logit2, interpolate_ratio2)
        framewise_logit2 = pad_framewise_output(framewise_logit2, frames_num)

        clipwise_output = (clipwise_output1 + clipwise_output2) / 2.0
        logit = (logit1 + logit2) / 2.0
        framewise_output = (framewise_output1 + framewise_output2) / 2.0
        framewise_logit = (framewise_logit1 + framewise_logit2) / 2.0

        return {
            "clipwise_output": clipwise_output,
            "logit": logit,
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
        }


# =============================================================================
# Quick sanity check when run as script
# =============================================================================
if __name__ == '__main__':
    model = SED(
        base_model_name=config.base_model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        in_channels=config.in_channels,
    )
    x = torch.randn([32, 3, 331, 224])
    output = model(x)
    print(output)
