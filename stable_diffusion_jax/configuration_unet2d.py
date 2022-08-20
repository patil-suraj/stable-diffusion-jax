from ast import Pass
from typing import Tuple

from transformers import PretrainedConfig


class UNet2DConfig(PretrainedConfig):
    def __init__(
        self,
        sample_size=None,
        in_channels=3,
        out_channels=3,
        center_input_sample=False,
        time_embedding_type="positional",
        freq_shift=0,
        flip_sin_to_cos=True,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(224, 448, 672, 896),
        layers_per_block=2,
        mid_block_scale_factor=1,
        downsample_padding=1,
        act_fn="silu",
        attention_head_dim=8,
        norm_num_groups=32,
        norm_eps=1e-5,
        dropout=0.1,
    ):
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center_input_sample = center_input_sample
        self.time_embedding_type = time_embedding_type
        self.freq_shift = freq_shift
        self.flip_sin_to_cos = flip_sin_to_cos
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.mid_block_scale_factor = mid_block_scale_factor
        self.downsample_padding = downsample_padding
        self.act_fn = act_fn
        self.attention_head_dim = attention_head_dim
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.dropout = dropout

        super().__init__()
