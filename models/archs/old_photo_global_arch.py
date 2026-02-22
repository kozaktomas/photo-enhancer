"""Global restoration architecture (VAE encoder/decoder + mapping network).

Adapted from Microsoft "Bringing Old Photos Back to Life" (CVPR 2020).
Source: Global/models/networks.py + NonLocal_feature_mapping_model.py
License: MIT

InstanceNorm2d is used as in the original pretrained checkpoint.
Training-only code removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type="reflect", norm_layer=nn.InstanceNorm2d, dilation=1):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer, dilation)

    def _build_conv_block(self, dim, padding_type, norm_layer, dilation):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(dilation)]
        elif padding_type == "zero":
            p = dilation

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=dilation),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class GlobalGenerator_DCDCv2(nn.Module):
    """VAE encoder/decoder for global restoration.

    Used as both VAE_A (quality) and VAE_B (scratch) with the same architecture
    but different checkpoint weights.

    Pretrained checkpoint was trained with: ngf=64, k_size=4, n_downsampling=3,
    start_r=1, mc=64, spatio_size=64, feat_dim=0, InstanceNorm2d.
    """

    def __init__(
        self,
        input_nc=3,
        output_nc=3,
        ngf=64,
        k_size=4,
        n_downsampling=3,
        norm_layer=nn.InstanceNorm2d,
        padding_type="reflect",
        start_r=1,
        mc=64,
        spatio_size=64,
    ):
        super().__init__()
        activation = nn.ReLU(True)

        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, mc), kernel_size=7, padding=0),
            norm_layer(min(ngf, mc)),
            activation,
        ]

        # Plain downsampling (no ResBlocks)
        for i in range(start_r):
            mult = 2**i
            in_ch = min(ngf * mult, mc)
            out_ch = min(ngf * mult * 2, mc)
            encoder += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=2, padding=1),
                norm_layer(out_ch),
                activation,
            ]

        # Downsampling with ResBlocks
        for i in range(start_r, n_downsampling - 1):
            mult = 2**i
            in_ch = min(ngf * mult, mc)
            out_ch = min(ngf * mult * 2, mc)
            encoder += [
                nn.Conv2d(in_ch, out_ch, kernel_size=k_size, stride=2, padding=1),
                norm_layer(out_ch),
                activation,
            ]
            encoder += [
                ResnetBlock(out_ch, padding_type=padding_type, norm_layer=norm_layer),
                ResnetBlock(out_ch, padding_type=padding_type, norm_layer=norm_layer),
            ]

        mult = 2 ** (n_downsampling - 1)
        ch = min(ngf * mult * 2, mc)

        if spatio_size == 32:
            encoder += [
                nn.Conv2d(ch, ch, kernel_size=k_size, stride=2, padding=1),
                norm_layer(ch),
                activation,
            ]
        if spatio_size == 64:
            encoder += [ResnetBlock(ch, padding_type=padding_type, norm_layer=norm_layer)]

        encoder += [ResnetBlock(ch, padding_type=padding_type, norm_layer=norm_layer)]
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        o_pad = 0 if k_size == 4 else 1
        mult = 2**n_downsampling
        dec_ch = min(ngf * mult, mc)

        decoder = [ResnetBlock(dec_ch, padding_type=padding_type, norm_layer=norm_layer)]

        if spatio_size == 32:
            decoder += [
                nn.ConvTranspose2d(
                    dec_ch,
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]
        if spatio_size == 64:
            decoder += [ResnetBlock(dec_ch, padding_type=padding_type, norm_layer=norm_layer)]

        for i in range(1, n_downsampling - start_r):
            mult = 2 ** (n_downsampling - i)
            in_ch = min(ngf * mult, mc)
            out_ch = min(int(ngf * mult / 2), mc)
            decoder += [
                ResnetBlock(in_ch, padding_type=padding_type, norm_layer=norm_layer),
                ResnetBlock(in_ch, padding_type=padding_type, norm_layer=norm_layer),
            ]
            decoder += [
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(out_ch),
                activation,
            ]

        for i in range(n_downsampling - start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            in_ch = min(ngf * mult, mc)
            out_ch = min(int(ngf * mult / 2), mc)
            decoder += [
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(out_ch),
                activation,
            ]

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(min(ngf, mc), output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input, flow="enc_dec"):
        if flow == "enc":
            return self.encoder(input)
        elif flow == "dec":
            return self.decoder(input)
        elif flow == "enc_dec":
            x = self.encoder(input)
            x = self.decoder(x)
            return x
        else:
            raise ValueError(f"Unknown flow: {flow}")


class NonLocalBlock2D_with_mask_Res(nn.Module):
    """Non-local block with mask-aware attention and residual ResBlocks.

    Matches the original pretrained checkpoint structure: W is a plain Conv2d
    (not wrapped in Sequential with BN), with 3 internal ResBlocks using
    InstanceNorm.
    """

    def __init__(self, in_channels, inter_channels=None, temperature=1.0, use_self=False):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels or in_channels

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.temperature = temperature
        self.use_self = use_self

        norm_layer = nn.InstanceNorm2d
        res_blocks = []
        for _ in range(3):
            res_blocks.append(
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    norm_layer=norm_layer,
                )
            )
        self.res_block = nn.Sequential(*res_blocks)

    def forward(self, x, mask):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f = f / self.temperature
        f_div_C = F.softmax(f, dim=2)

        # Mask processing
        tmp = 1 - mask
        mask_interp = F.interpolate(
            mask, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        mask_interp[mask_interp > 0] = 1.0
        mask_interp = 1 - mask_interp

        tmp = F.interpolate(tmp, (x.size(2), x.size(3)))
        mask_interp = mask_interp * tmp

        mask_expand = mask_interp.view(batch_size, 1, -1)
        mask_expand = mask_expand.repeat(1, x.size(2) * x.size(3), 1)

        if self.use_self:
            diag_idx = range(x.size(2) * x.size(3))
            mask_expand[:, diag_idx, diag_idx] = 1.0

        f_div_C = mask_expand * f_div_C

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, x.size(2), x.size(3))
        W_y = self.W(y)
        W_y = self.res_block(W_y)

        # Combine: blend original and attention-output using mask
        full_mask = mask_interp.repeat(1, self.in_channels, 1, 1)
        z = full_mask * x + (1 - full_mask) * W_y
        return z


class Mapping_Model_with_mask_2(nn.Module):
    """Feature translation mapping network with scratch mask.

    Maps VAE_A encoder features to VAE_B decoder space using a non-local
    attention block that is aware of scratch mask regions.

    Pretrained checkpoint structure: before_NL (4 conv layers 64→512),
    NL (single non-local block at 512), after_NL (6 ResBlocks + 4 conv
    layers 512→64).
    """

    def __init__(self, nc=64, mc=512):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        activation = nn.ReLU(True)

        # before_NL: 4 conv layers progressively increasing channels
        before = []
        n_up = 4
        for i in range(n_up):
            ic = min(nc * (2**i), mc)
            oc = min(nc * (2 ** (i + 1)), mc)
            before += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        self.before_NL = nn.Sequential(*before)

        # NL: single non-local attention block
        self.NL = NonLocalBlock2D_with_mask_Res(mc, inter_channels=mc)

        # after_NL: 6 ResBlocks + downscale convs back to nc
        after = []
        for _ in range(6):
            after.append(ResnetBlock(mc, padding_type="reflect", norm_layer=norm_layer))
        for i in range(n_up - 1):
            ic = min(nc * (2 ** (n_up - i)), mc)
            oc = min(nc * (2 ** (n_up - 1 - i)), mc)
            after += [nn.Conv2d(ic, oc, 3, 1, 1), norm_layer(oc), activation]
        after += [nn.Conv2d(nc * 2, nc, 3, 1, 1)]
        self.after_NL = nn.Sequential(*after)

    def forward(self, input, mask):
        x = self.before_NL(input)
        x = self.NL(x, mask)
        x = self.after_NL(x)
        return x
