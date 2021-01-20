import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from torch import Tensor
import collections
from typing import Callable, Any, Optional, List

class MMPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=kernel_size, ceil_mode=True)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, ceil_mode=True)

    def forward(self, x):
        return self.avgpool(x) + self.maxpool(x)



class CNN6(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.embed = kwargs.get('embed', 256)

        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.init_bn = nn.BatchNorm1d(inputdim)
        self.features = nn.Sequential(
            _block(1, 64),
            nn.MaxPool2d(2),
            _block(64, 128),
            nn.MaxPool2d(2),
            _block(128, self.embed),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Dropout(0.2, True),
        )
        self.outputlayer = nn.Linear(self.embed, outputdim)
        self.attention = nn.Linear(self.embed, self.embed)

    def forward(self, x):
        x = self.init_bn(x.transpose(1, 2)).transpose(1, 2)
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        #Attention
        w = torch.softmax(self.attention(x), dim=1)
        x = (x * w).sum(1) / (w.sum(1) + 1e-7)
        return self.outputlayer(x)




class CNN10(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.embed = kwargs.get('embed',512)


        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.init_bn = nn.BatchNorm1d(inputdim)
        self.features = nn.Sequential(
            _block(1, 64),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(64, 128),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(128, 256),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(256, self.embed),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.outputlayer = nn.Linear(self.embed, outputdim)

    def forward(self, x):
        x = self.init_bn(x.transpose(1,2)).transpose(1,2)
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = x.mean(1) + x.max(1)[0]
        return self.outputlayer(x)

class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class LinearSoftPool(nn.Module):
    """LinearSoftPool

    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:

        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050

    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class CRNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = LinearSoftPool()
        self.outputlayer = nn.Linear(256, outputdim)

    def forward(self, x, upsample=True):
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        if upsample:
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)
        return decision, decision_time


class CNNVAD(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.vad_model = CRNN(inputdim, 2)
        self.classifier = CNN10(inputdim, outputdim, **kwargs)

    def forward(self, x):
        _, pred = self.vad_model(x) #Prediction is 2 labels: Speech (1) and Non-Speech (0)
        pred = pred[..., 1, None]
        return self.classifier(pred * x)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )

# necessary for backwards compatibility of MobilenetV2
ConvBNReLU = ConvBNActivation



class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        inputdim: int = None, # Dummy parameter for compatibility
        outputdim: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNReLU(1, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, outputdim),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = x.unsqueeze(1)
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
