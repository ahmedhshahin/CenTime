"""
Neural network model.

Implements the neural network model. 

Author: Ahmed H. Shahin
Date: 31/8/2023
"""
from typing import Optional, Tuple

import torch
from torch import nn


class NonlinearActivation(nn.Module):
    """
    Implements nonlinear activation layers.

    Args:
        act_type (str): Type of activation function. Default is 'relu'.
    """

    def __init__(self, act_type: str = "relu"):
        super().__init__()
        if act_type == "relu":
            self.nonlin = nn.ReLU(inplace=True)
        elif act_type == "leaky_relu":
            self.nonlin = nn.LeakyReLU(inplace=True, negative_slope=0.01)
        else:
            raise ValueError(
                f"Invalid activation type: {act_type}. Please select from relu, leaky_relu."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the NonlinearActivation module.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the nonlinear activation.
        """
        return self.nonlin(input_tensor)


class Normalization(nn.Module):
    """
    Implements normalization layers.

    Args:
        norm_type (str): Type of normalization. Either 'bn' for BatchNorm or 'in' for
          InstanceNorm. Default is 'bn'.
        in_channels (int): Number of input channels. Required for both 'bn' and 'in'.

    Raises:
        ValueError: If an invalid normalization type is provided.
    """

    def __init__(self, norm_type: str = "bn", in_channels: int = None):
        super().__init__()
        if norm_type == "bn":
            self.norm = nn.BatchNorm3d(in_channels)
        elif norm_type == "in":
            self.norm = nn.InstanceNorm3d(in_channels)
        else:
            raise ValueError(
                f"Invalid normalization type: {norm_type}. Please select from bn and in."
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the normalization layer.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return self.norm(input_tensor)


class ResidualBlock(nn.Module):
    """
    Implements a residual block with three convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        act (str): Activation function type ('relu', 'leaky_relu', etc.)
        norm (str, optional): Normalization type ('bn' for BatchNorm or 'in' for
          InstanceNorm). Default is 'in'.
    """

    def __init__(
        self, in_channels: int, out_channels: int, act: str, norm: Optional[str] = "in"
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm1 = Normalization(norm_type=norm, in_channels=out_channels // 2)
        self.nonlin1 = NonlinearActivation(act_type=act)

        self.conv2 = nn.Conv3d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = Normalization(norm_type=norm, in_channels=out_channels // 2)
        self.nonlin2 = NonlinearActivation(act_type=act)

        self.conv3 = nn.Conv3d(
            out_channels // 2,
            out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm3 = Normalization(norm_type=norm, in_channels=out_channels // 2)

        self.conv_res = nn.Conv3d(
            in_channels,
            out_channels // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.nonlin_out = NonlinearActivation(act_type=act)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the residual block.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """
        x_res = self.conv_res(input_tensor)

        input_tensor = self.conv1(input_tensor)
        input_tensor = self.norm1(input_tensor)
        input_tensor = self.nonlin1(input_tensor)

        input_tensor = self.conv2(input_tensor)
        input_tensor = self.norm2(input_tensor)
        input_tensor = self.nonlin2(input_tensor)

        input_tensor = self.conv3(input_tensor)
        input_tensor = self.norm3(input_tensor)

        input_tensor = torch.cat((input_tensor, x_res), dim=1)
        input_tensor = self.nonlin_out(input_tensor)

        return input_tensor


class Model(nn.Module):
    """
    Implements a 3D Convolutional Neural Network with optional clinical data integration.

    Args:
        in_filters (int, optional): Number of input filters for the first convolutional
          layer. Default is 16.
        filters (Tuple[int], optional): Tuple of output filters for each residual block.
          Default is (16).
        out_filters (int, optional): Number of output filters for the last convolutional
          layer. Default is 128.
        act (str, optional): Activation function type ('relu', 'leaky_relu', etc.).
          Default is 'relu'.
        norm (str, optional): Normalization type ('bn' for BatchNorm or 'in' for
          InstanceNorm). Default is 'in'.
        n_classes (int, optional): Number of output classes. Default is 2.
        n_clinical_data (int, optional): Number of clinical features.
          Default is -1 (no clinical data).
    """

    def __init__(
        self,
        in_filters: int = 16,
        filters: Tuple[int] = (16,),
        out_filters: int = 128,
        act: str = "relu",
        norm: str = "in",
        n_classes: int = 2,
        n_clinical_data: int = -1,
    ) -> None:
        super().__init__()
        self.n_clinical_data = n_clinical_data
        layers = []

        # Initial convolutional layer
        layers.append(
            nn.Conv3d(1, in_filters, kernel_size=1, stride=1, padding=0, bias=False)
        )
        layers.append(Normalization(norm, in_filters))
        layers.append(NonlinearActivation(act))

        # Residual blocks
        for i, filt in enumerate(filters):
            layers.append(
                ResidualBlock(in_filters if i == 0 else filters[i - 1], filt, act=act)
            )
            layers.append(
                nn.Conv3d(
                    filt, filt, kernel_size=3, stride=2, padding=2, dilation=2, bias=False
                )
            )
            layers.append(Normalization(norm, filt))
            layers.append(NonlinearActivation(act))
        self.layers = nn.Sequential(*layers)

        self.out_conv = nn.Conv3d(
            filters[-1], out_filters, kernel_size=1, stride=1, padding=0, bias=False
        )
        out_feats = 256 // (2 ** len(filters))

        # Conditional layers based on the presence of clinical data
        if n_clinical_data > 0:
            self.fc_imaging = nn.Sequential(
                nn.Linear(out_filters * out_feats**3, 32),
                nn.BatchNorm1d(32),
                NonlinearActivation(act),
            )
            self.fc_clinical = nn.Sequential(
                nn.Linear(n_clinical_data, 32),
                nn.BatchNorm1d(32),
                NonlinearActivation(act),
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                NonlinearActivation(act),
            )
            self.fc_output = nn.Linear(32 + 32, n_classes)
        else:
            self.fc_mean = nn.Linear(out_filters * out_feats**3, n_classes)

    def forward(
        self, x_img: torch.Tensor, clinical_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            x_img (torch.Tensor): Input image tensor.
            clinical_data (torch.Tensor, optional): Input clinical data tensor.
              Default is None.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x_img = self.layers(x_img)
        x_img = self.out_conv(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        if self.n_clinical_data > 0:
            x_clinical = self.fc_clinical(clinical_data)
            x_img = self.fc_imaging(x_img)
            x_data = torch.cat((x_img, x_clinical), dim=1)
            x_data = self.fc_output(x_data)
            return x_data

        x_img = self.fc_mean(x_img)
        return x_img
