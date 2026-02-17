import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


def swish(x):
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if in_channels != out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_out = None

    def forward(self, x_in):
        x = self.norm1(x_in)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.conv_out is not None:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)  # b, hw, c
        k = k.reshape(b, c, h * w)  # b, c, hw
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w)  # b, c, hw
        w_ = w_.permute(0, 2, 1)  # b, hw, hw
        h_ = torch.bmm(v, w_)  # b, c, hw
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, emb_dim, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(codebook_size, emb_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / codebook_size,
            1.0 / codebook_size,
        )

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.emb_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices):
        return self.embedding(indices)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        nf,
        emb_dim,
        ch_mult,
        num_res_blocks,
        resolution,
        attn_resolutions,
    ):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions

        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(in_channels, nf, 3, 1, 1))

        # body
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2

        # end
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, 3, 1, 1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3

        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        blocks = []
        # initial conv
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, 3, 1, 1))

        # mid
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))

        # body (reverse of encoder)
        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2

        # end
        blocks.append(normalize(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, 3, 1, 1))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        return out


class VQAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        nf,
        ch_mult,
        quantizer="nearest",
        res_blocks=2,
        attn_resolutions=None,
        codebook_size=1024,
        emb_dim=256,
        beta=0.25,
    ):
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = [16]

        self.in_channels = 3
        self.nf = nf
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim

        self.encoder = Encoder(
            self.in_channels,
            nf,
            emb_dim,
            ch_mult,
            self.n_blocks,
            img_size,
            attn_resolutions,
        )
        self.quantize = VectorQuantizer(codebook_size, emb_dim, beta=beta)
        self.generator = Generator(
            nf,
            emb_dim,
            ch_mult,
            self.n_blocks,
            img_size,
            attn_resolutions,
        )
