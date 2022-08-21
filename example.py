import jax

from transformers import FlaxCLIPTextModel, CLIPTokenizer

from stable_diffusion_jax import StableDiffusionPipeline, UNet2D, AutoencoderKL, PNDMScheduler
from stable_diffusion_jax.pipeline_stable_diffusion import InferenceState
from stable_diffusion_jax.convert_diffusers_to_jax import convert_diffusers_to_jax

# convert diffusers checkpoint to jax
pt_path = "path_to_diffusers_pt_ckpt"
fx_path = "save_path"
convert_diffusers_to_jax(pt_path, fx_path)

# inference with jax
clip_model, clip_params = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", _do_init=False)
unet, unet_params = UNet2D.from_pretrained(f"{fx_path}/unet", _do_init=False)
vae, vae_params = AutoencoderKL.from_pretrained(f"{fx_path}/vae", _do_init=False)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler = PNDMScheduler()


state = InferenceState(text_encoder_params=clip_params, unet_params=unet_params, vae_params=vae_params)
# for single GPU 
state = jax.tree_map(lambda arr: jax.device_put(arr, jax.devices()[0]), state)


pipe = StableDiffusionPipeline(text_encoder=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler, vae=vae)

p = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
input_ids = tokenizer(p, padding="max_length", truncation=True, max_length=77, return_tensors="jax").input_ids
uncond_input_ids = tokenizer([""], padding="max_length", truncation=True, max_length=77, return_tensors="jax").input_ids
rng = jax.random.PRNGKey(42)

image = pipe.sample(
    input_ids, uncond_input_ids, prng_seed=rng, inference_state=state, guidance_scale=7.5, num_inference_steps=50
)

# with jit
sample = jax.jit(pipe.sample, static_argnums=4)
image = sample(input_ids, uncond_input_ids, prng_seed=rng, inference_state=state, guidance_scale=7.5, num_inference_steps=50)