import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqgan_arch import Fuse_sft_block, ResBlock, VQAutoEncoder


class TransformerSALayer(nn.Module):
    def __init__(
        self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            batch_first=False,
        )
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None):
        # Self-attention with positional encoding
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        # Feed-forward
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class CodeFormer(VQAutoEncoder):
    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=None,
        fix_modules=None,
    ):
        if connect_list is None:
            connect_list = ["32", "64", "128", "256"]
        if fix_modules is None:
            fix_modules = ["quantize", "generator"]

        super().__init__(
            512,
            64,
            [1, 2, 2, 4, 4, 8],
            quantizer="nearest",
            res_blocks=2,
            attn_resolutions=[16],
            codebook_size=codebook_size,
            emb_dim=256,
        )

        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2

        self.position_emb = nn.Parameter(torch.zeros(latent_size, dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)

        self.ft_layers = nn.Sequential(
            *[
                TransformerSALayer(
                    embed_dim=dim_embd,
                    nhead=n_head,
                    dim_mlp=self.dim_mlp,
                    dropout=0.0,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, codebook_size, bias=False),
        )

        self.channels = {
            "16": 512,
            "32": 256,
            "64": 256,
            "128": 128,
            "256": 128,
        }

        # SFT fusion layers (single dict matches v0.1.0 checkpoint layout)
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        # ---------- Encoder ----------
        enc_feat_dict = {}
        out_list = [self.connect_list]  # noqa: F841
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if isinstance(block, ResBlock):
                if x.shape[-1] in [int(f) for f in self.connect_list]:
                    enc_feat_dict[str(x.shape[-1])] = x.clone()

        lq_feat = x

        # ---------- Transformer ----------
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb

        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)

        # Predict codebook indices
        logits = self.idx_pred_layer(query_emb)  # (HW, B, codebook_size)
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)

        quant_feat = self.quantize.get_codebook_entry(top_idx.reshape(-1))
        quant_feat = (
            quant_feat.reshape(
                lq_feat.shape[0],
                lq_feat.shape[2],
                lq_feat.shape[3],
                -1,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        if code_only:
            return top_idx.reshape(lq_feat.shape[0], -1), soft_one_hot

        # ---------- Generator with SFT fusion ----------
        if detach_16:
            quant_feat = quant_feat.detach()

        if adain:
            quant_feat = self._adaptive_instance_normalization(
                quant_feat,
                lq_feat,
            )

        x = quant_feat
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if isinstance(block, ResBlock):
                if x.shape[-1] in [int(f) for f in self.connect_list]:
                    f_size = str(x.shape[-1])
                    if w > 0:
                        x = self.fuse_convs_dict[f_size](
                            enc_feat_dict[f_size].detach(),
                            x,
                            w,
                        )

        out = x
        return out, logits, lq_feat

    @staticmethod
    def _adaptive_instance_normalization(content_feat, style_feat):
        size = content_feat.size()  # noqa: F841
        style_mean = style_feat.mean([2, 3], keepdim=True)
        style_std = style_feat.std([2, 3], keepdim=True) + 1e-8
        content_mean = content_feat.mean([2, 3], keepdim=True)
        content_std = content_feat.std([2, 3], keepdim=True) + 1e-8
        normalized = (content_feat - content_mean) / content_std
        return normalized * style_std + style_mean
