from functools import partial
from typing import Tuple
import math

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict

from transformers.modeling_flax_utils import FlaxPreTrainedModel

from .configuration_vae import Unet2DConfig

class SinusoidalEmbedding(nn.Module):
  dim: int = 32
  
  @nn.compact
  def __call__(self, inputs):
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = inputs[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
    return emb


class TimestepEmbedding(nn.Module):
  dim: int = 32
  
  @nn.compact
  def __call__(self, inputs):
    time_dim = self.dim * 4
  
    se = SinusoidalEmbedding(self.dim)(inputs)
    x = nn.Dense(time_dim)(se)
    x = nn.gelu(x)
    x = nn.Dense(time_dim)(x)
    return x


class Upsample(nn.Module):
  in_channels: int
  with_conv: bool
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    if self.with_conv:
      self.conv = nn.Conv(
          self.in_channels,
          kernel_size=(3, 3),
          strides=(1, 1),
          padding=((1, 1), (1, 1)),
          dtype=self.dtype,
      )

  def __call__(self, hidden_states):
    batch, height, width, channels = hidden_states.shape
    hidden_states = jax.image.resize(
        hidden_states,
        shape=(batch, height * 2, width * 2, channels),
        method="nearest",
    )
    if self.with_conv:
      hidden_states = self.conv(hidden_states)
    return hidden_states


class Downsample(nn.Module):
  in_channels: int
  with_conv: bool
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    if self.with_conv:
      self.conv = nn.Conv(
          self.in_channels,
          kernel_size=(3, 3),
          strides=(2, 2),
          padding="VALID",
          dtype=self.dtype,
      )

  def __call__(self, hidden_states):
    if self.with_conv:
      pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
      hidden_states = jnp.pad(hidden_states, pad_width=pad)
      hidden_states = self.conv(hidden_states)
    else:
      hidden_states = nn.avg_pool(hidden_states,
                                  window_shape=(2, 2),
                                  strides=(2, 2),
                                  padding="VALID")
    return hidden_states


class ResnetBlock(nn.Module):
    pass


class SpatialTransformer(nn.Module):
    pass


class AttentionBlock(nn.Module):
    pass


class UNetMidBlock2D(nn.Module):
    pass


class DownBlock2D(nn.Module):
    pass


class UpBlock2D(nn.Module):
    pass


class AttnUpBlock2D(nn.Module):
    pass


class AttnDownBlock2D(nn.Module):
    pass


class UNet2DPretrainedModel(FlaxPreTrainedModel):
    pass


class UNet2D(UNet2DPretrainedModel):
    pass