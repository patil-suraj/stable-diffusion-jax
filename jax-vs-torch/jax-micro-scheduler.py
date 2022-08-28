import copy
import jax

from stable_diffusion_jax.scheduling_pndm import PNDMSchedulerState
num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
# assert device_type.startswith("TPU"), "Available device is not a TPU, please select TPU from Edit > Notebook settings > Hardware accelerator"

import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from stable_diffusion_jax import PNDMScheduler

# Local checkout
flax_path = "/sddata/sd-v1-4-flax"

dtype = jnp.float32
scheduler = PNDMScheduler.from_config(f"{flax_path}/scheduler")
initial_state = scheduler.state.state_dict.copy()

# Using jax.debug.print() makes it crash :()
def mini_sample(
    latents: jnp.ndarray,
    noise: jnp.ndarray,
    scheduler_state_dict: dict,
    num_inference_steps: int = 50,
    break_after: int = 50,
):
    scheduler_state = PNDMSchedulerState.from_state_dict(scheduler_state_dict)
    scheduler_state = scheduler.set_timesteps(scheduler_state, latents.shape, num_inference_steps, offset=1)
    scheduler_state_dict = scheduler_state.state_dict

    def loop_body(step, args):
        latents, scheduler_state_dict = args
        t = jnp.array(scheduler_state_dict["timesteps"])[step]

        # compute the previous noisy sample x_t -> x_t-1
        latents, scheduler_state_dict = scheduler.step(scheduler_state_dict, noise, t, latents)
        latents = latents["prev_sample"]
        return latents, scheduler_state_dict #, t

    n = min(len(scheduler_state.timesteps), break_after)

    latents, scheduler_state_dict = jax.lax.fori_loop(0, n, loop_body, (latents, scheduler_state_dict))
    # for step in range(n):
    #     latents, scheduler_state_dict, t = loop_body(step, (latents, scheduler_state_dict))
    #     print(f"{step} [{t}]: {latents}")

    return latents, scheduler_state_dict

p_sample = jax.pmap(mini_sample, in_axes=(0, 0, None), static_broadcasted_argnums=(3,4))

# Run tests on a single device: "tpu" or "cpu".
# Running iteratively is very slow, so `shard` and `replicate` anyway.

# devices = jax.devices("cpu")[:1]
devices = jax.devices()[:1]
print(devices)

# # Test the first few items to begin with
# latents = jnp.array([[0.34100962, -1.0947237, -1.778018, 0.43691084]])
# noise   = jnp.array([[0.14349212, -1.2879468, -1.0073951, -1.8549157]])
# latents = jax.device_put_sharded([latents], devices)
# noise   = jax.device_put_sharded([noise], devices)

# scheduler_state_dict = initial_state.copy()
# latents, _ = mini_sample(latents, noise, scheduler_state_dict, 50, 51)


for step in range(50+1):
    latents = jnp.array([[0.34100962, -1.0947237, -1.778018, 0.43691084]])
    noise   = jnp.array([[0.14349212, -1.2879468, -1.0073951, -1.8549157]])
    latents = jax.device_put_sharded([latents], devices)
    noise   = jax.device_put_sharded([noise], devices)

    scheduler_state_dict = initial_state.copy()
    latents, _ = p_sample(latents, noise, scheduler_state_dict, 50, step+1)
    print(f"Step: {step}: {latents}")
