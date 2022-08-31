import torch

from diffusers import StableDiffusionPipeline

device = "cpu"

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True
).to(device)

num_inference_steps = 50
torch.manual_seed(42)

latents = torch.tensor([[0.34100962, -1.0947237, -1.778018, 0.43691084]])
noise = [torch.randn((1, 4)) for _ in range(num_inference_steps+1)]

scheduler = pipe.scheduler
scheduler.set_timesteps(num_inference_steps, offset=1)

with torch.no_grad():
    for i, (t, noise_sample) in enumerate(zip(scheduler.timesteps, noise)):
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_sample, t, latents)["prev_sample"]
        # print(f"Current noise: {noise_sample.numpy()}")
        print(f"{i} [{t}]: {latents.detach().numpy()}")