import pickle
import torch

from pathlib import Path
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, UNet2DConditionModel

device = "cuda"

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True
).to(device)

# Generate latents and save them

seed = 7667
torch.manual_seed(seed)
latents = torch.randn(
    (1, pipe.unet.in_channels, 512 // 8, 512 // 8),
    generator=None,
    device=pipe.device,
)

Path("tensors").mkdir(exist_ok=True)
with open(f'tensors/latents_{seed}', 'wb') as f:
    pickle.dump(latents.to("cpu"), f)

# Generate text embeddings

prompt = ["Cute puppy"]
text_input = pipe.tokenizer(
    prompt,
    padding="max_length",
    max_length=pipe.tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
)
text_embeddings = pipe.text_encoder(text_input.input_ids.to(pipe.device))[0]

with open(f'tensors/embeddings_{seed}', 'wb') as f:
    pickle.dump(text_embeddings.detach().to("cpu").numpy(), f)

num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps, offset=1)

with torch.no_grad():
    for i, t in tqdm(enumerate(pipe.scheduler.timesteps)):
        # predict the noise residual
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_embeddings)["sample"]

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents)["prev_sample"]

    latents = 1 / 0.18215 * latents

with open(f"tensors/latents_{seed}_final", "wb") as f:
    pickle.dump(latents.to("cpu"), f)
