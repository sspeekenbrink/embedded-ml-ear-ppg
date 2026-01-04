from dataclasses import dataclass
from torch import nn


def create_pooling_layer(pooling_type: str, kernel_size: int = 2):
    """
    Creates a 1D pooling layer based on the specified type.

    Args:
        pooling_type: Type of pooling layer to create
        kernel_size: Kernel size (default: 2)

    Returns:
        nn.Module: The appropriate pooling layer

    Supported pooling types:
        - "max": MaxPool1d
        - "avg": AvgPool1d
        - "lp_2": LPPool1d with norm_type=2
    """
    pooling_type = pooling_type.lower()

    if pooling_type == "max":
        return nn.MaxPool1d(kernel_size)
    elif pooling_type == "avg":
        return nn.AvgPool1d(kernel_size)
    elif pooling_type == "lp_2":
        return nn.LPPool1d(2.0, kernel_size)
    else:
        supported_types = ["max", "avg", "lp_2"]
        raise ValueError(
            f"Unsupported pooling type: {pooling_type}. Choose from: {supported_types}"
        )


@dataclass(frozen=True)
class CNNConfig:
    base_channels: int = 32
    num_blocks: int = 4
    kernel_size: int = 5
    dilation: int = 1
    pool_every: int = 2
    use_batchnorm: bool = True
    activation: str = "silu"
    pooling: str = "max"
    double_every: int = 2


def build_cnn(cfg: CNNConfig) -> nn.Module:
    layers = []
    in_ch = 1
    upscaler_count = 0

    for i in range(cfg.num_blocks):
        num_doubles = i // cfg.double_every
        out_ch = cfg.base_channels * (2**num_doubles)

        conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=cfg.kernel_size,
            padding=((cfg.kernel_size - 1) // 2) * cfg.dilation,
            dilation=cfg.dilation,
        )
        layers.append(conv)

        if cfg.use_batchnorm:
            layers.append(nn.BatchNorm1d(out_ch))

        if cfg.activation.lower() == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif cfg.activation.lower() == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            raise ValueError(
                f"Unsupported activation function: {cfg.activation}. Choose 'silu' or 'relu'."
            )

        if (i + 1) % cfg.pool_every == 0:
            layers.append(create_pooling_layer(cfg.pooling, kernel_size=2))
            upscaler_count += 1

        in_ch = out_ch

    if upscaler_count > 0:
        layers.append(nn.Upsample(scale_factor=2**upscaler_count, mode="nearest"))
    layers.append(nn.Conv1d(in_ch, 1, kernel_size=1))

    return nn.Sequential(*layers)


class PPGPeakDetectorCNN(nn.Module):
    """
    A static, pre-defined CNN architecture for PPG peak detection.
    This serves as a baseline model if not using grid search.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class DilatedCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 32, kernel_size=5, padding=2, dilation=1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(32, 64, kernel_size=3, padding=1, dilation=1)
        self.pool2 = nn.MaxPool1d(2)

        self.dilated_conv1 = ConvBlock(64, 128, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv2 = ConvBlock(128, 128, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv3 = ConvBlock(128, 128, kernel_size=3, padding=4, dilation=4)
        self.dilated_conv4 = ConvBlock(128, 128, kernel_size=3, padding=8, dilation=8)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = ConvBlock(128, 64, kernel_size=3, padding=1, dilation=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ConvBlock(64, 32, kernel_size=3, padding=1, dilation=1)

        self.out_conv = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool1(x)
        x = self.enc2(x)
        x = self.pool2(x)

        x = self.dilated_conv1(x)
        x = self.dilated_conv2(x)
        x = self.dilated_conv3(x)
        x = self.dilated_conv4(x)

        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)

        logits = self.out_conv(x)
        return logits


@dataclass(frozen=True)
class DilatedCNNConfig:
    """Configurable Dilated CNN similar to DilatedCNN, for grid exploration."""

    in_channels: int = 1
    enc1_channels: int = 32
    enc1_kernel: int = 5
    enc2_channels: int = 64
    enc2_kernel: int = 3
    dec1_channels: int = 64
    dec1_kernel: int = 5
    dec2_channels: int = 32
    dec2_kernel: int = 5
    bottleneck_channels: int = 128
    num_dilated_layers: int = 4
    dilated_kernel: int = 3
    base_dilation: int = 2
    out_channels: int = 1
    activation: str = "silu"
    pooling: str = "max"


def build_dilated_cnn(cfg: DilatedCNNConfig) -> nn.Module:
    layers = []

    enc1_pad = (cfg.enc1_kernel - 1) // 2
    layers.append(
        ConvBlock(
            cfg.in_channels,
            cfg.enc1_channels,
            kernel_size=cfg.enc1_kernel,
            padding=enc1_pad,
            dilation=1,
            activation=cfg.activation,
        )
    )
    layers.append(create_pooling_layer(cfg.pooling, kernel_size=2))

    enc2_pad = (cfg.enc2_kernel - 1) // 2
    layers.append(
        ConvBlock(
            cfg.enc1_channels,
            cfg.enc2_channels,
            kernel_size=cfg.enc2_kernel,
            padding=enc2_pad,
            dilation=1,
            activation=cfg.activation,
        )
    )
    layers.append(create_pooling_layer(cfg.pooling, kernel_size=2))

    in_ch = cfg.enc2_channels
    for i in range(cfg.num_dilated_layers):
        dilation = cfg.base_dilation**i
        pad = dilation * (cfg.dilated_kernel - 1) // 2
        layers.append(
            ConvBlock(
                in_ch if i == 0 else cfg.bottleneck_channels,
                cfg.bottleneck_channels,
                kernel_size=cfg.dilated_kernel,
                padding=pad,
                dilation=dilation,
                activation=cfg.activation,
            )
        )

    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    dec1_pad = (cfg.dec1_kernel - 1) // 2
    layers.append(
        ConvBlock(
            cfg.bottleneck_channels,
            cfg.dec1_channels,
            kernel_size=cfg.dec1_kernel,
            padding=dec1_pad,
            dilation=1,
            activation=cfg.activation,
        )
    )
    layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
    dec2_pad = (cfg.dec2_kernel - 1) // 2
    layers.append(
        ConvBlock(
            cfg.dec1_channels,
            cfg.dec2_channels,
            kernel_size=cfg.dec2_kernel,
            padding=dec2_pad,
            dilation=1,
            activation=cfg.activation,
        )
    )
    layers.append(nn.Conv1d(cfg.dec2_channels, cfg.out_channels, kernel_size=1))

    return nn.Sequential(*layers)


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        dilation,
        activation="silu",
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)

        if activation.lower() == "silu":
            self.activation = nn.SiLU(inplace=True)
        elif activation.lower() == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(
                f"Unsupported activation function: {activation}. Choose 'silu' or 'relu'."
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)
