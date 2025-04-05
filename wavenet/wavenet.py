import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation,
                              bias=False, **kwargs)

    def forward(self, x):
        if self.padding:
            return self.conv(x)[:, :, :-self.padding]
        return self.conv(x)


class GatedActivationUnit(nn.Module):
    def __init__(self):
        super(GatedActivationUnit, self).__init__()

    def forward(self, x):
        return torch.tanh(x) * torch.sigmoid(x)


class WaveNet(nn.Module):
    def __init__(self, num_block=4, num_layer=10, class_dim=256, residual_dim=32, dilation_dim=128, skip_dim=256,
                 kernel_size=2, bias=False):
        super(WaveNet, self).__init__()

        self.start_conv = nn.Conv1d(in_channels=class_dim, out_channels=residual_dim, kernel_size=1, bias=bias)

        self.stack = nn.ModuleList()
        for _ in range(num_block):
            dilation = 1
            for _ in range(num_layer):
                self.stack.append(
                    nn.Sequential(
                        CausalConv1d(in_channels=residual_dim, out_channels=dilation_dim,
                                     kernel_size=kernel_size, dilation=dilation
                                     ),
                        GatedActivationUnit(),
                        nn.Conv1d(in_channels=dilation_dim, out_channels=residual_dim, kernel_size=1, bias=bias)
                    )
                )
                dilation *= 2

        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=residual_dim, out_channels=skip_dim, kernel_size=1, bias=bias),
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_dim, out_channels=class_dim, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        residual = self.start_conv(x)
        skips = torch.zeros_like(residual)

        for layer in self.stack:
            skip = layer(residual)
            residual = residual + skip
            skips = skips + skip
        logits = self.end_conv(skips)
        return logits
