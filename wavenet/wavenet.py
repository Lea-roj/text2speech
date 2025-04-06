import torch
import torch.nn as nn

from config import Option

opt = Option()


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


class ResidualBlock(nn.Module):
    def __init__(self, res_dim, dil_dim, skip_dim, kernel_size, dilation, bias=False):
        super().__init__()
        self.causal = CausalConv1d(res_dim, dil_dim, kernel_size, dilation)
        self.gate = GatedActivationUnit()
        self.res_proj = nn.Conv1d(dil_dim, res_dim, 1, bias=bias)
        self.skip_proj = nn.Conv1d(dil_dim, skip_dim, 1, bias=bias)

    def forward(self, x):
        gated = self.gate(self.causal(x))
        res = self.res_proj(gated)
        skip = self.skip_proj(gated)
        return x + res, skip


class WaveNet(nn.Module):
    def __init__(self, num_block=opt.num_block,
                 num_layer=opt.num_layer,
                 class_dim=opt.num_class,
                 residual_dim=opt.residual_dim,
                 dilation_dim=opt.dilation_dim,
                 skip_dim=opt.skip_dim,
                 kernel_size=opt.kernel_size,
                 bias=opt.bias
                 ):
        super(WaveNet, self).__init__()

        self.start_conv = nn.Conv1d(in_channels=class_dim, out_channels=residual_dim, kernel_size=1, bias=bias)

        self.stack = nn.ModuleList()
        for _ in range(num_block):
            dilation = 1
            for _ in range(num_layer):
                self.stack.append(
                    ResidualBlock(
                        res_dim=residual_dim,
                        dil_dim=dilation_dim,
                        skip_dim=skip_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        bias=bias
                    )
                )
                dilation *= 2

        self.end_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_dim, out_channels=skip_dim, kernel_size=1, bias=bias),
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_dim, out_channels=class_dim, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        residual = self.start_conv(x)
        skips = 0

        for block in self.stack:
            residual, skip = block(residual)
            skips = skips + skip if isinstance(skips, torch.Tensor) else skip

        logits = self.end_conv(skips)
        return logits
