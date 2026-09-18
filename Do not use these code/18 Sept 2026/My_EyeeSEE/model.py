
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 3,
                 base_filters: int = 64, dropout: float = 0.1):
        super().__init__()
        f = base_filters
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,      f * 2)
        self.down2 = Down(f * 2,  f * 4)
        self.down3 = Down(f * 4,  f * 8)
        self.down4 = Down(f * 8,  f * 16)
        self.drop  = nn.Dropout2d(dropout)
        self.up1   = Up(f * 16,  f * 8)
        self.up2   = Up(f * 8,   f * 4)
        self.up3   = Up(f * 4,   f * 2)
        self.up4   = Up(f * 2,   f)
        self.outc  = nn.Conv2d(f, n_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.drop(self.down4(x4))
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred  = F.softmax(pred, dim=1)
        oh    = F.one_hot(target, pred.size(1)).permute(0, 3, 1, 2).float()
        inter = (pred * oh).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
        return 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()


class CombinedLoss(nn.Module):
    """
    0.5 * Dice  +  0.5 * weighted CrossEntropy
    Weights:  background=0.1   disc=1.5   cup=3.0
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self._w   = torch.tensor([0.2, 1.5, 6.0])
        self.ce   = nn.CrossEntropyLoss(weight=self._w)

    def to(self, device):
        super().to(device)
        self._w = self._w.to(device)
        self.ce = nn.CrossEntropyLoss(weight=self._w)
        return self

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.ce(pred, target)


def calculate_dice(pred, target) -> float:
    import numpy as np
    p = pred.astype(bool)
    t = target.astype(bool)
    return float((2.0 * (p & t).sum() + 1e-5) / (p.sum() + t.sum() + 1e-5))
