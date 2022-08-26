# Loop test

Procedure:
- Run `torch-micro-loop.py` in a Linux box for reference. It will save the initial state (latents and text embeddings), and the final latents.
- Run `jax-micro-loop.py` in a TPU. It will load the initial state, run the loop and compare the results.

Notes:
- "cpu" and "tpu" devices in a TPU give the same results unless I'm doing something wrong.
- I fixed a small difference in the scheduler (offset=1), but its impact is low. I think it looks correct, I run it with some fixed inputs and the results were the same. I'll repeat with a more complex text.
- Runing a full `for` loop instead of a `fori_loop` produces double the difference. I don't undernstand the reason why.
- In another test, I looped without involving the scheduler (the output from the unet was the input for the next cycle) and the differences were much smaller. They did accumulate as iterations grow (I saved intermediate results every 10 steps).

Summary of differences:

* Torch cuda vs JAX TPU
```
Max: 3.999680519104004
Mean: 0.32840099930763245
```

* Torch CPU (in TPU device) vs JAX TPU
```
Max: 2.8781819343566895
Mean: 0.3067624568939209
```

* Torch CPU (Linux) vs JAX TPU
```
Max: 2.8606467247009277
Mean: 0.30655285716056824
```

If I hardcode step 20 in the scheduler call, the max difference goes down to `~1`.