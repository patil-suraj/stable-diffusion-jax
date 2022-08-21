## TODOs:

- [x] Finish implementing the `UNet2D` model in `modeling_unte2d.py`. Port weights of any existing LDM unet from diffusers and verify equivalence. I've added the skleton of modules that we need to implement in the file.
- [x] Adapt the `PNDMScheduler` from `diffusers` for JAX: Use `jnp` arrays and make it stateless.
- [x] Add the KL module from (here)[https://github.dev/CompVis/stable-diffusion] in `modeling_vae.py` file. For inference we don't really need it, but would be nice to have for completeness. Port the weights of any existing KL VAE and verify equivalence.
- [x] Add an inference loop in `pipeline_stabel_diffusion`. We should able to `jit`/`pmap` the loop to deploy on TPUs.
