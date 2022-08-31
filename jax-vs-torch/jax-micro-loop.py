import jax
import jax.numpy as jnp
import os

device = "cuda"
run_parallel = True            # Set to False for interactive / debugging
dtype = jnp.float32
enable_x64_timesteps = True

if enable_x64_timesteps:
    # Experimental: enable int64 for timestep.
    # This will perform some computations in float64, then we'll truncate to float32.
    jax.config.update("jax_enable_x64", True)

if device == "cpu":
    # Make sure we really use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    jax.config.update('jax_platform_name', 'cpu')

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
# assert device_type.startswith("TPU"), "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

from flax.jax_utils import replicate
from flax.training.common_utils import shard

import numpy as np

from PIL import Image
from stable_diffusion_jax.scheduling_pndm import PNDMSchedulerState
from stable_diffusion_jax import (
    AutoencoderKL,
    PNDMScheduler,
    UNet2D
)

from tqdm import tqdm
import pickle

# Local checkout
flax_path = "/sddata/sd-v1-4-flax"
tensors_path = "/sddata/sd-tests/tensors"

unet, unet_params = UNet2D.from_pretrained(f"{flax_path}/unet", _do_init=False, dtype=dtype)
scheduler = PNDMScheduler.from_config(f"{flax_path}/scheduler")
initial_state = scheduler.state.state_dict.copy()

# Using jax.debug.print() makes it crash :()
def mini_sample(
    text_embeddings: jnp.ndarray,
    latents: jnp.ndarray,
    unet_params,
    scheduler_state_dict: dict,
    num_inference_steps: int = 50,
    break_after: int = 51,
):
    scheduler_state = PNDMSchedulerState.from_state_dict(scheduler_state_dict)
    scheduler_state = scheduler.set_timesteps(scheduler_state, latents.shape, num_inference_steps, offset=1)
    scheduler_state_dict = scheduler_state.state_dict

    def loop_body(step, args):
        latents, scheduler_state_dict = args
        t = jnp.array(scheduler_state_dict["timesteps"], dtype=jnp.int64)[step]
        timestep = jnp.broadcast_to(t, latents.shape[0])
        
        # predict the noise residual
        noise_pred = unet(
            latents, timestep, encoder_hidden_states=text_embeddings, params=unet_params
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state_dict = scheduler.step(scheduler_state_dict, noise_pred, t, latents)
        latents = jnp.array(latents["prev_sample"], dtype=jnp.float32)
        return latents, scheduler_state_dict

    n = min(len(scheduler_state.timesteps), break_after)
    if run_parallel:
        latents, scheduler_state_dict = jax.lax.fori_loop(0, n, loop_body, (latents, scheduler_state_dict))
    else:
        for step in range(n):
            latents, scheduler_state_dict = loop_body(step, (latents, scheduler_state_dict))
            print(f"{step}: {latents[0, 0, 0, 1]}")

    return latents, scheduler_state_dict

p_sample = jax.pmap(mini_sample, in_axes=(0, 0, 0, None), static_broadcasted_argnums=(4,5))

# Run tests on a single device
devices = jax.devices(device)[:1]

def read_latents():
    with open(f'{tensors_path}/latents_7667_cuda', 'rb') as f:
        latents = pickle.load(f)

    latents = jnp.array(latents)
    latents = jnp.transpose(latents, (0, 2, 3, 1))
    return latents

latents = read_latents()

with open(f'{tensors_path}/embeddings_7667', 'rb') as f:
    text_embeddings = pickle.load(f)
text_embeddings = jnp.array(text_embeddings)

if run_parallel:
    latents = jax.device_put_sharded([latents], devices)
    unet_params = replicate(unet_params, devices)
    text_embeddings = jax.device_put_sharded([text_embeddings], devices)

num_inference_steps = 50
sample_fn = p_sample if run_parallel else mini_sample
for step in [50]:
    latents = read_latents()
    if run_parallel:
        latents = jax.device_put_sharded([latents], devices)

    scheduler_state_dict = initial_state.copy()
    latents, _ = sample_fn(text_embeddings, latents, unet_params, scheduler_state_dict, num_inference_steps, step+1)

    latents = latents[0]        # unshard
    slice = latents[0, 0, 0, :] if run_parallel else latents[0, 0, :]
    print(f"Step: {step}: {slice}")


# scheduler_state_dict = initial_state.copy()
# latents, _ = p_sample(text_embeddings, latents, unet_params, scheduler_state_dict, num_inference_steps, 51)

latents = 1 / 0.18215 * latents

vae, vae_params = AutoencoderKL.from_pretrained(f"{flax_path}/vae", _do_init=False, dtype=dtype)
images = vae.decode(latents, params=vae_params)

# convert images to PIL images
images = images / 2 + 0.5
images = jnp.clip(images, 0, 1)
images = (images * 255).round().astype("uint8")
images = np.asarray(images).reshape((1, 512, 512, 3))

pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save(f"jax_{device}_fixes.png")


with open(f"{tensors_path}/latents_7667_cuda_final", "rb") as f:
    torch_latents = pickle.load(f)
torch_latents = jnp.transpose(jnp.array(torch_latents), (0, 2, 3, 1))

assert torch_latents.shape == latents.shape, "Wrong shapes"

print(f"Sum: {jnp.sum(jnp.abs((latents - torch_latents)))}")
print(f"Max: {jnp.max(jnp.abs((latents - torch_latents)))}")
print(f"Mean: {jnp.mean(jnp.abs((latents - torch_latents)))}")
