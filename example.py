import jax
import jax.numpy as jnp
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPTokenizer, FlaxCLIPTextModel, CLIPConfig

from stable_diffusion_jax import (
    AutoencoderKL,
    InferenceState,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2D,
    StableDiffusionSafetyCheckerModel,
)


# Local checkout until weights are available in the Hub
flax_path = "/sddata/sd-v1-4-flax"

num_samples = 8
num_inference_steps = 50
guidance_scale = 7.5

devices = jax.devices()[:1]

# inference with jax
dtype = jnp.bfloat16
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained(
    "openai/clip-vit-large-patch14", _do_init=False, dtype=dtype
)
unet, unet_params = UNet2D.from_pretrained(f"{flax_path}/unet", _do_init=False, dtype=dtype)
vae, vae_params = AutoencoderKL.from_pretrained(f"{flax_path}/vae", _do_init=False, dtype=dtype)
safety_model, safety_model_params = StableDiffusionSafetyCheckerModel.from_pretrained(f"{flax_path}/safety_checker", _do_init=False, dtype=dtype)

config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

latents_shape = (
    num_samples,
    unet.config.sample_size,
    unet.config.sample_size,
    unet.config.in_channels,
)

scheduler = PNDMScheduler.from_config(f"{flax_path}/scheduler")
scheduler_state = scheduler.set_timesteps(
    scheduler.state,
    latents_shape,
    num_inference_steps = num_inference_steps,
    offset = 1,
)

# create inference state and replicate it across all TPU devices
inference_state = InferenceState(
    text_encoder_params=clip_params,
    unet_params=unet_params,
    vae_params=vae_params,
    scheduler_state=scheduler_state,
)
inference_state = replicate(inference_state, devices=devices)


# create pipeline
pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vae=vae)


# prepare inputs
p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"

input_ids = tokenizer(
    [p] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
uncond_input_ids = tokenizer(
    [""] * num_samples, padding="max_length", truncation=True, max_length=77, return_tensors="jax"
).input_ids
prng_seed = jax.random.PRNGKey(42)

# shard inputs and rng
# Simply use shard if using the default devices
input_ids = jax.device_put_sharded([input_ids], devices)
uncond_input_ids = jax.device_put_sharded([uncond_input_ids], devices)
prng_seed = jax.random.split(prng_seed, len(devices))

# pmap the sample function
sample = jax.pmap(pipe.sample, static_broadcasted_argnums=(4,))

# sample images
images = sample(
    input_ids,
    uncond_input_ids,
    prng_seed,
    inference_state,
    guidance_scale,
)


# convert images to PIL images
images = images / 2 + 0.5
images = jnp.clip(images, 0, 1)
images = (images * 255).round().astype("uint8")
images = np.asarray(images).reshape((num_samples, 512, 512, 3))

pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("example.png")
