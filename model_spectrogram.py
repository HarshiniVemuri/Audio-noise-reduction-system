"""
Spectrogram-based CNN Autoencoder (2D U-Net) for Audio Noise Reduction.
Input : (B, 1, n_mels, T) log-mel spectrogram
Output: (B, 1, n_mels, T) denoised spectrogram
"""

import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TransposeBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1, out_padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding, out_padding),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class SpectrogramAutoencoder(nn.Module):
    """2D spectrogram-based U-Net autoencoder."""

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock2D(1, 32)
        self.enc2 = ConvBlock2D(32, 64)
        self.enc3 = ConvBlock2D(64, 128)
        self.enc4 = ConvBlock2D(128, 256)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.GELU(),
        )

        self.dec4 = TransposeBlock2D(256 + 256, 128)
        self.dec3 = TransposeBlock2D(128 + 128, 64)
        self.dec2 = TransposeBlock2D(64 + 64, 32)
        self.dec1 = TransposeBlock2D(32 + 32, 16)

        self.out = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([b, e4], dim=1))
        d3 = self.dec3(torch.cat([_match2d(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([_match2d(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([_match2d(d2, e1), e1], dim=1))

        return self.out(_match2d(d1, x))


def _match2d(x, target):
    if x.shape != target.shape:
        x = torch.nn.functional.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
    return x
