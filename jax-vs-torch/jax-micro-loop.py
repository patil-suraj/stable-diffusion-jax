import jax
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert device_type.startswith("TPU"), "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D
)

from tqdm import tqdm
import pickle

# Local checkout
flax_path = "sd-v1-4-flax"

dtype = jnp.float32
unet, unet_params = UNet2D.from_pretrained(f"{flax_path}/unet", _do_init=False, dtype=dtype)
scheduler = PNDMScheduler.from_config(f"{flax_path}/scheduler")

# Using jax.debug.print() makes it crash :()
def mini_sample(
    text_embeddings: jnp.ndarray,
    latents: jnp.ndarray,
    unet_params,
    num_inference_steps: int = 50,
):
    scheduler.set_timesteps(num_inference_steps, offset=1)

    def loop_body(step, latents):
        t = jnp.array(scheduler.timesteps)[step]
        timestep = jnp.broadcast_to(t, latents.shape[0])

#         jax.debug.print(f"🤯 {t}")
        
        # predict the noise residual
        noise_pred = unet(
            latents, timestep, encoder_hidden_states=text_embeddings, params=unet_params
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    latents = jax.lax.fori_loop(0, num_inference_steps, loop_body, latents)
    return latents

p_sample = jax.pmap(mini_sample, static_broadcasted_argnums=(3,))

# Run tests on a single device: "tpu" or "cpu".
# Running iteratively is very slow, so `shard` and `replicate` anyway.

devices = jax.devices("tpu")[:1]

unet_params = replicate(unet_params, devices)
with open('tensors/latents_7667', 'rb') as f:
    latents = pickle.load(f)

latents = jnp.array(latents)
latents = jnp.transpose(latents, (0, 2, 3, 1))

with open('tensors/embeddings_7667', 'rb') as f:
    text_embeddings = pickle.load(f)

text_embeddings = jnp.array(text_embeddings)

text_embeddings = jax.device_put_sharded([text_embeddings], devices)
latents = jax.device_put_sharded([latents], devices)

latents = p_sample(text_embeddings, latents, unet_params, 50)

latents = 1 / 0.18215 * latents

with open("tensors/latents_7667_final", "rb") as f:
    torch_latents = pickle.load(f)
torch_latents = jnp.transpose(jnp.array(torch_latents), (0, 2, 3, 1))

print(f"Max: {jnp.max(jnp.abs((latents - torch_latents)))}")
print(f"Mean: {jnp.mean(jnp.abs((latents - torch_latents)))}")
