import torch
import torch.nn as nn
from ops.modules import MSDeformAttn


class AVSTransformerEncoderLayer(nn.Module):
    def __init__(self, dim=256, ffn_dim=2048, dropout=0.1, num_levels=3, num_heads=8, num_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(dim, num_levels, num_heads, num_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        # ffn
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(
            src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # ffn
        src = self.ffn(src)
        return src


class AVSTransformerEncoder(nn.Module):
    def __init__(self, num_layers, layer, *args, **kwargs) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [AVSTransformerEncoderLayer(**layer) for i in range(num_layers)]
        )

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        out = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for layer in self.layers:
            out = layer(out, pos, reference_points, spatial_shapes,
                        level_start_index, padding_mask)
        return out, reference_points


class AVSTransformerDecoderLayer(nn.Module):
    def __init__(self, dim=256, num_heads=8, ffn_dim=2048, dropout=0.1, num_levels=3, num_points=4, *args, **kwargs) -> None:
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)

        # cross attention
        # self.cross_attn = MSDeformAttn(dim, num_levels, num_heads, num_points)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)

        # ffn
        self.linear1 = nn.Linear(dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(dim)

    def ffn(self, src):
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src

    def forward(self, query, src, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        out1 = self.self_attn(query, query, query)[0]
        query = query + self.dropout1(out1)
        query = self.norm1(query)
        # cross attention
        out2 = self.cross_attn(
            query, src, src, key_padding_mask=padding_mask)[0]
        query = query + self.dropout2(out2)
        query = self.norm2(query)
        # ffn
        query = self.ffn(query)
        return query


class AVSTransformerDecoder(nn.Module):
    def __init__(self, num_layers, layer, *args, **kwargs) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [AVSTransformerDecoderLayer(**layer) for i in range(num_layers)]
        )

    def forward(self, query, src, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        out = query
        outputs = []
        for layer in self.layers:
            out = layer(out, src, reference_points, spatial_shapes,
                        level_start_index, padding_mask)
            outputs.append(out)
        return outputs


class AVSTransformer(nn.Module):
    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__()

        self.encoder = AVSTransformerEncoder(**encoder)
        self.decoder = AVSTransformerDecoder(**decoder)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, query, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        memory, reference_points = self.encoder(
            src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask)
        outputs = self.decoder(query, memory, reference_points,
                               spatial_shapes, level_start_index, padding_mask)
        return memory, outputs


def build_transformer(type, **kwargs):
    if type == 'AVSTransformer':
        return AVSTransformer(**kwargs)
    else:
        raise ValueError
