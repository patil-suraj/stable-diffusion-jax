import inspect
import jax
import jax.numpy as jnp
import flax.linen as nn
from scheduling_pndm import PNDMScheduler
from PIL import Image

from transformers import CLIPTokenizer, FlaxCLIPTextModel


class FlaxLDMTextToImagePipeline:
    def __init__(self, vqvae, clip, tokenizer, unet, scheduler):
        scheduler = scheduler.set_format("jax")
        self.vqvae = vqvae
        self.clip = clip
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def __call__(
        self,
        prompt,
        batch_size=1,
        prng_seed=None,
        eta=0.0,
        guidance_scale=1.0,
        num_inference_steps=50,
        output_type="pil",
    ):

        # eta corresponds to η in paper and should be between [0, 1]
        batch_size = len(prompt)

        # get unconditional embeddings for classifier free guidance
        if guidance_scale != 1.0:
            uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="jax")
        else:
            uncond_input = None

        # get prompt text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="jax")

        self.scheduler.set_timesteps(num_inference_steps)

        def _forward(
            self,
            text_input,
            uncond_input=None,
        ):
            if guidance_scale != 1.0:
                uncond_embeddings = self.clip(uncond_input.input_ids)[0]

            text_embeddings = self.clip(text_input.input_ids)[0]

            latents = jax.random.normal(
                prng_seed,
                shape=(text_input.input_ids.shape[0], self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
                dtype=jnp.float32
            )

            # TODO(Nathan) - make the following work with JAX
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_kwrags = {}
            if not accepts_eta:
                extra_kwrags["eta"] = eta

            for t in self.scheduler.timesteps:
                if guidance_scale == 1.0:
                    # guidance_scale of 1 means no guidance
                    latents_input = latents
                    context = text_embeddings
                else:
                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    latents_input = jnp.concatenate([latents] * 2)
                    context = jnp.concatenate([uncond_embeddings, text_embeddings])

                # predict the noise residual
                noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
                # perform guidance
                if guidance_scale != 1.0:
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_kwrags)["prev_sample"]

            # TODO wait until vqvae is ready in FLAX and then correct that here
            # image = self.vqvae.decode(latents)
            # scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = latents

            return image

        # or jax.pmap
        jit_forward = jax.jit(_forward)

        image = jit_forward(text_input, uncond_input)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}

# that's the official CLIP model and tokenizer Stable-diffusion uses
# see: https://github.com/CompVis/stable-diffusion/blob/ce05de28194041e030ccfc70c635fe3707cdfc30/configs/stable-diffusion/v1-inference.yaml#L70
# and https://github.com/CompVis/stable-diffusion/blob/ce05de28194041e030ccfc70c635fe3707cdfc30/ldm/modules/encoders/modules.py#L137
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
clip_model = FlaxCLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")


class DummyUnet(nn.Module):
    in_channels = 3
    sample_size = 1

    @nn.compact
    def __call__(self, latents_input, t, encoder_hidden_states):
        return {"sample": latents_input + 1}

unet = DummyUnet()
scheduler = PNDMScheduler()


pipeline = FlaxLDMTextToImagePipeline(vqvae=None, clip=clip_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)

# now running the pipeline should work more or less which it doesn't at the moment @Nathan
key = jax.random.PRNGKey(0)

prompt = "A painting of a squirrel eating a burger"
images = pipeline([prompt], prng_seed=key, num_inference_steps=50, eta=0.3, guidance_scale=6)["sample"]