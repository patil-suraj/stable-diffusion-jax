from ast import Pass
from typing import Tuple

from transformers import PretrainedConfig

class Unet2DConfig(PretrainedConfig):
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
    ):
        pass