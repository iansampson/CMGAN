import torch
from torch import nn, einsum
import torch.nn.functional as F
import torchaudio

from einops import rearrange
from einops.layers.torch import Rearrange


# Replace lucidrains implementation
# with torchaudio.models.Conformer –
# both to appease the CoreML converter
# and for faster performance
class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.conformer = torchaudio.models.Conformer(input_dim=dim,
                                                     num_heads=heads,
                                                     ffn_dim=dim * ff_mult,
                                                     num_layers=1,
                                                     depthwise_conv_kernel_size=conv_kernel_size,
                                                     dropout=attn_dropout)

    def forward(self, x, mask = None):
        lengths = torch.full((1, x.shape[0]), x.shape[1]).squeeze()
        return self.conformer(x, lengths)[0]
