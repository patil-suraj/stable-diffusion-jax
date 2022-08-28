import pickle
import torch

from diffusers import StableDiffusionPipeline

device = "cpu"

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True
).to(device)

# Generate latents and save them

latents = torch.tensor([[0.34100962, -1.0947237, -1.778018, 0.43691084]])
noise   = torch.tensor([[0.14349212, -1.2879468, -1.0073951, -1.8549157]])

num_inference_steps = 50

scheduler = pipe.scheduler
scheduler.set_timesteps(num_inference_steps, offset=1)

with torch.no_grad():
    for i, t in enumerate(scheduler.timesteps):
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise, t, latents)["prev_sample"]
        print(f"{i} [{t}]: {latents.detach().numpy()}")