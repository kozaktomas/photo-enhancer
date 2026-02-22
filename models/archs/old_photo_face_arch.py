"""SPADE face enhancement generator architecture.

Adapted from Microsoft "Bringing Old Photos Back to Life" (CVPR 2020).
Source: Face_Enhancement/models/networks/ (generator, architecture, normalization, encoder)
License: MIT

SynchronizedBatchNorm2d replaced with nn.BatchNorm2d. Training-only code removed.
Consolidated from multiple source files into a single module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SPADE(nn.Module):
    """Spatially-adaptive normalization (SPADE) layer."""

    def __init__(self, norm_nc, label_nc, nhidden=128):
        super().__init__()

        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    """ResNet block with SPADE normalization."""

    def __init__(self, fin, fout, semantic_nc):
        super().__init__()
        self.learned_shortcut = fin != fout
        fmiddle = min(fin, fout)

        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        return x_s + dx


class ConvEncoder(nn.Module):
    """Encoder that extracts a mean and variance vector from the input image."""

    def __init__(self, input_nc=3, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        kw = 3
        pw = 1

        self.layer1 = norm_layer(nn.Conv2d(input_nc, ngf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ngf, ngf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ngf * 2, ngf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ngf * 4, ngf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ngf * 8, ngf * 8, kw, stride=2, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)

        self.fc_mu = nn.Linear(ngf * 8 * 4 * 4, 256)
        self.fc_var = nn.Linear(ngf * 8 * 4 * 4, 256)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class SPADEGenerator(nn.Module):
    """SPADE-based face enhancement generator.

    Takes degraded face image as semantic input and produces enhanced output.
    Architecture: initial FC → 7 SPADE ResBlocks with progressive upsampling.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, semantic_nc=3, use_vae=False):
        super().__init__()
        self.use_vae = use_vae
        nf = ngf

        self.sw = self.sh = 4  # Spatial size of first feature map

        if use_vae:
            self.fc = nn.Linear(256, 16 * nf * self.sw * self.sh)
        else:
            self.fc = nn.Conv2d(semantic_nc, 16 * nf, kernel_size=3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, semantic_nc)
        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, semantic_nc)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, semantic_nc)
        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, semantic_nc)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, semantic_nc)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, semantic_nc)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, semantic_nc)

        self.conv_img = nn.Conv2d(nf, output_nc, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, input, z=None):
        seg = input

        if self.use_vae:
            x = self.fc(z)
            x = x.view(-1, 16 * 64, self.sh, self.sw)  # ngf=64
        else:
            x = F.interpolate(seg, size=(self.sh, self.sw), mode="bilinear", align_corners=False)
            x = self.fc(x)

        x = self.head_0(x, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.up(x)
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x
