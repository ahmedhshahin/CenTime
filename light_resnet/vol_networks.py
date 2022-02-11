# This file is part of Dynamic Affine Feature Map Transform (DAFT).
#
# DAFT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DAFT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DAFT. If not, see <https://www.gnu.org/licenses/>.
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from light_resnet.base import BaseModel
from light_resnet.vol_blocks import ConvBnReLU, DAFTBlock, FilmBlock, ResBlock


class HeterogeneousResNet(BaseModel):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, n_basefilters=4, normalization='bn') -> None:
        super().__init__()

        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum, normalization=normalization)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, normalization=normalization)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2, normalization=normalization)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2, normalization=normalization)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2, normalization=normalization)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular=None):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class ConcatHNN1FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters + ndim_non_img, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return out


class ConcatHNN2FC(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ):

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        layers = [
            ("fc1", nn.Linear(8 * n_basefilters + ndim_non_img, bottleneck_dim)),
            # ("dropout", nn.Dropout(p=0.5, inplace=True)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(bottleneck_dim, n_outputs)),
        ]
        self.fc = nn.Sequential(OrderedDict(layers))

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, tabular), dim=1)
        out = self.fc(out)

        return out


class ConcatHNNMCM(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
        bottleneck_dim: int = 7,
    ) -> None:

        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Linear(ndim_non_img, bottleneck_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(8 * n_basefilters + bottleneck_dim, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        tab_transformed = self.mlp(tabular)
        tab_transformed = self.relu(tab_transformed)
        out = torch.cat((out, tab_transformed), dim=1)
        out = self.fc(out)

        return out


class InteractiveHNN(BaseModel):
    """
    adapted version of Duanmu et al. (MICCAI, 2020)
    https://link.springer.com/chapter/10.1007%2F978-3-030-59713-9_24
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        ndim_non_img: int = 15,
    ) -> None:

        super().__init__()

        # ResNet
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.block4 = ResBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

        layers = [
            ("aux_base", nn.Linear(ndim_non_img, 8, bias=False)),
            ("aux_relu", nn.ReLU()),
            # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(8, n_basefilters, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

        self.aux_2 = nn.Linear(n_basefilters, n_basefilters, bias=False)
        self.aux_3 = nn.Linear(n_basefilters, 2 * n_basefilters, bias=False)
        self.aux_4 = nn.Linear(2 * n_basefilters, 4 * n_basefilters, bias=False)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)

        attention = self.aux(tabular)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block1(out)

        attention = self.aux_2(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block2(out)

        attention = self.aux_3(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block3(out)

        attention = self.aux_4(attention)
        batch_size, n_channels = out.size()[:2]
        out = torch.mul(out, attention.view(batch_size, n_channels, 1, 1, 1))
        out = self.block4(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class FilmHNN(BaseModel):
    """
    adapted version of Perez et al. (AAAI, 2018)
    https://arxiv.org/abs/1709.07871
    """

    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8
        self.blockX = FilmBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.blockX(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class DAFT(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        normalization='bn',
        filmblock_args: Optional[Dict[Any, Any]] = None,
        handle_missing = 'zeros',
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.handle_missing = handle_missing
        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum, normalization=normalization)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum, normalization=normalization)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2, normalization=normalization)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2, normalization=normalization)  # 8
        self.block4 = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, normalization=normalization, **filmblock_args)  # 4
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(8 * n_basefilters, n_outputs)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out, tabular)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class DAFT_MoE(BaseModel):
    def __init__(
        self,
        in_channels: int,
        n_outputs: int,
        bn_momentum: float = 0.1,
        n_basefilters: int = 4,
        filmblock_args: Optional[Dict[Any, Any]] = None,
    ) -> None:
        super().__init__()

        if filmblock_args is None:
            filmblock_args = {}

        self.split_size = 4 * n_basefilters
        self.conv1 = ConvBnReLU(in_channels, n_basefilters, bn_momentum=bn_momentum)
        self.pool1 = nn.MaxPool3d(2, stride=2)  # 32
        self.block1 = ResBlock(n_basefilters, n_basefilters, bn_momentum=bn_momentum)
        self.block2 = ResBlock(n_basefilters, 2 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 16
        self.block3 = ResBlock(2 * n_basefilters, 4 * n_basefilters, bn_momentum=bn_momentum, stride=2)  # 8

        self.blockX1 = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.blockX2 = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.blockX3 = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4
        self.blockX4 = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4

        self.classifier = DAFTBlock(4 * n_basefilters, 8 * n_basefilters, bn_momentum=bn_momentum, **filmblock_args)  # 4

        self.pool_1 = nn.AdaptiveAvgPool3d(1)
        self.pool_2 = nn.AdaptiveAvgPool3d(1)
        self.pool_3 = nn.AdaptiveAvgPool3d(1)
        self.pool_4 = nn.AdaptiveAvgPool3d(1)

        self.pool_cls = nn.AdaptiveAvgPool3d(1)

        self.fc1 = nn.Linear(8 * n_basefilters, n_outputs)
        self.fc2 = nn.Linear(8 * n_basefilters, n_outputs)
        self.fc3 = nn.Linear(8 * n_basefilters, n_outputs)
        self.fc4 = nn.Linear(8 * n_basefilters, n_outputs)

        self.fc_cls = nn.Linear(8 * n_basefilters, 4)

    @property
    def input_names(self) -> Sequence[str]:
        return ("image", "tabular")

    @property
    def output_names(self) -> Sequence[str]:
        return ("logits",)

    def forward(self, image, tabular):
        out = self.conv1(image)
        out = self.pool1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out1 = self.blockX1(out, tabular)
        out2 = self.blockX2(out, tabular)
        out3 = self.blockX3(out, tabular)
        out4 = self.blockX4(out, tabular)

        scores = self.classifier(out, tabular)

        out1 = self.pool_1(out1).view(out.size(0), -1)
        out2 = self.pool_2(out2).view(out.size(0), -1)
        out3 = self.pool_3(out3).view(out.size(0), -1)
        out4 = self.pool_4(out4).view(out.size(0), -1)
        scores = self.pool_cls(scores).view(out.size(0), -1)

        out1 = self.fc1(out1)
        out2 = self.fc2(out2)
        out3 = self.fc3(out3)
        out4 = self.fc4(out4)

        scores = nn.functional.softmax(self.fc_cls(scores), dim=1)
        agg = torch.cat([out1,out2,out3,out4], dim=1)
        agg = agg.multiply(scores).sum(axis=1)
        return agg, scores