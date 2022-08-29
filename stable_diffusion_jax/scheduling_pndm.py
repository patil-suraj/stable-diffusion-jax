# Copyright 2022 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

from typing import Tuple

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce. :param alpha_bar: a lambda that takes an argument t
    from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return jnp.array(betas, dtype=jnp.float32)


@dataclass
class PNDMSchedulerState:
    betas: jnp.array
    num_train_timesteps: int

    num_inference_steps = None
    _timesteps = None   #jnp.arange(0, num_train_timesteps)[::-1].copy()
    _offset = 0
    prk_timesteps = None
    plms_timesteps = None
    timesteps = None
    
    # For now we only support F-PNDM, i.e. the runge-kutta method
    # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
    # mainly at formula (9), (12), (13) and the Algorithm 2.
    pndm_order = 4

    # running values
    model_output = jnp.array([])
    counter = 0
    sample = jnp.array([])
    ets = jnp.array([])

    @property
    def alphas(self) -> jnp.array:
        return 1.0 - self.betas

    @property
    def alphas_cumprod(self) -> jnp.array:
        return jnp.cumprod(self.alphas, axis=0)

    @property
    def state_dict(self) -> dict:
        return {
            "betas": self.betas,
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "_timesteps": self._timesteps,
            "_offset": self._offset,
            "prk_timesteps": self.prk_timesteps,
            "plms_timesteps": self.plms_timesteps,
            "timesteps": self.timesteps,
            "model_output": self.model_output,
            "counter": self.counter,
            "sample": self.sample,
            "ets": self.ets,
        }

    @classmethod
    def from_state_dict(cls, dict):
        # TODO: verify keys
        state = cls(betas=dict.get("betas"), num_train_timesteps=dict.get("num_train_timesteps"))
        for k, v in dict.items():
            state.__setattr__(k, v)
        return state

    
class PNDMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        tensor_format="np",
        skip_prk_steps=True,
    ):

        if beta_schedule == "linear":
            betas = jnp.linspace(beta_start, beta_end, num_train_timesteps, dtype=jnp.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = jnp.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=jnp.float32) ** 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        # Temporarily stored here, should be returned
        self.state = PNDMSchedulerState(betas, num_train_timesteps)

        self.tensor_format = tensor_format
        self.set_format(tensor_format=tensor_format)

    def set_timesteps(
        self,
        state: PNDMSchedulerState,
        shape: Tuple,
        num_inference_steps: int,
        offset=0,
    ) -> PNDMSchedulerState:
        state.num_inference_steps = num_inference_steps
        # self._timesteps = list(
        #     range(0, self.config.num_train_timesteps, self.config.num_train_timesteps // num_inference_steps)
        # )
        state._timesteps = jnp.arange(
            0, self.config.num_train_timesteps, self.config.num_train_timesteps // num_inference_steps
        )
        state._offset = offset
        # self._timesteps = [t + self._offset for t in self._timesteps]
        state._timesteps = state._timesteps + state._offset

        if self.config.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            state.prk_timesteps = jnp.array([])
            # self.plms_timesteps = list(reversed(self._timesteps[:-1] + self._timesteps[-2:-1] + self._timesteps[-1:]))
            state.plms_timesteps = jnp.concatenate(
                (state._timesteps[:-1], state._timesteps[-2:-1], state._timesteps[-1:])
            )[::-1]
        else:
            prk_timesteps = state._timesteps[-state.pndm_order :].repeat(2) + jnp.tile(
                jnp.array([0, self.config.num_train_timesteps // num_inference_steps // 2]), state.pndm_order
            )
            state.prk_timesteps = prk_timesteps[:-1].repeat(2)[1:-1][::-1]
            state.plms_timesteps = state._timesteps[:-3][::-1]

        timesteps = jnp.concatenate((state.prk_timesteps, state.plms_timesteps))
        state.timesteps = jnp.array(timesteps, dtype=jnp.int32)

        # Will be zeros, not really empty
        state.ets = jnp.empty((4,) + shape)
        state.sample = jnp.empty(shape)
        state.model_output = jnp.empty(shape)
        state.counter = 0
        self.set_format(tensor_format=self.tensor_format)

        return state

    def step(
        self,
        state_dict: dict,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ):
        return self.step_plms(state_dict=state_dict, model_output=model_output, timestep=timestep, sample=sample)
        # if self.config.skip_prk_steps:
        #     return self.step_plms(state_dict=state_dict, model_output=model_output, timestep=timestep, sample=sample)
        
        # return jnp.where(
        #     state.counter < len(state.prk_timesteps),
        #     self.step_prk(state_dict=state_dict, model_output=model_output, timestep=timestep, sample=sample),
        #     self.step_plms(state_dict=state_dict, model_output=model_output, timestep=timestep, sample=sample)
        # )

    # def step_prk(
    #     self,
    #     state_dict: PNDMSchedulerState,
    #     model_output: jnp.ndarray,
    #     timestep: int,
    #     sample: jnp.ndarray,
    # ):
    #     """
    #     Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
    #     solution to the differential equation.
    #     """
    #     diff_to_prev = 0 if state.counter % 2 else self.config.num_train_timesteps // state.num_inference_steps // 2
    #     prev_timestep = max(timestep - diff_to_prev, state.prk_timesteps[-1])
    #     timestep = state.prk_timesteps[state.counter // 4 * 4]

    #     if state.counter % 4 == 0:
    #         state.cur_model_output += 1 / 6 * model_output
    #         state.ets.append(model_output)
    #         state.cur_sample = sample
    #     elif (state.counter - 1) % 4 == 0:
    #         state.cur_model_output += 1 / 3 * model_output
    #     elif (state.counter - 2) % 4 == 0:
    #         state.cur_model_output += 1 / 3 * model_output
    #     elif (state.counter - 3) % 4 == 0:
    #         model_output = state.cur_model_output + 1 / 6 * model_output
    #         state.cur_model_output = 0

    #     # cur_sample should not be `None`
    #     cur_sample = state.cur_sample if state.cur_sample is not None else sample

    #     prev_sample = self._get_prev_sample(state, cur_sample, timestep, prev_timestep, model_output)
    #     state.counter += 1

    #     return {"prev_sample": prev_sample}

    def step_plms(
        self,
        state_dict: dict,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ):
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.
        """
        state = PNDMSchedulerState.from_state_dict(state_dict)
        if not self.config.skip_prk_steps and len(state.ets) < 3:
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        prev_timestep = timestep - self.config.num_train_timesteps // state.num_inference_steps
        prev_timestep = jnp.where(prev_timestep > 0, prev_timestep, 0)

        # if state.counter != 1:
        #     state.ets.append(model_output)
        # else:
        #     prev_timestep = timestep
        #     timestep = timestep + self.config.num_train_timesteps // state.num_inference_steps
        prev_timestep = jnp.where(state.counter == 1, timestep, prev_timestep)
        timestep = jnp.where(state.counter == 1, timestep + self.config.num_train_timesteps // state.num_inference_steps, timestep)

        # if len(state.ets) == 1 and state.counter == 0:
        #     model_output = model_output
        #     state.cur_sample = sample
        # elif len(state.ets) == 1 and state.counter == 1:
        #     model_output = (model_output + state.ets[-1]) / 2
        #     sample = state.cur_sample
        #     state.cur_sample = None
        # elif len(state.ets) == 2:
        #     model_output = (3 * state.ets[-1] - state.ets[-2]) / 2
        # elif len(state.ets) == 3:
        #     model_output = (23 * state.ets[-1] - 16 * state.ets[-2] + 5 * state.ets[-3]) / 12
        # else:
        #     model_output = (1 / 24) * (55 * state.ets[-1] - 59 * state.ets[-2] + 37 * state.ets[-3] - 9 * state.ets[-4])

        def counter_0(state_dict):
            ets = state_dict["ets"]
            ets = ets.at[0].set(model_output)
            state_dict["ets"] = ets

            state_dict["sample"] = sample
            state_dict["model_output"] = model_output
            return state_dict

        def counter_1(state_dict):
            state_dict["model_output"] = (model_output + state_dict["ets"][0]) / 2
            return state_dict

        def counter_2(state_dict):
            ets = state_dict["ets"]
            ets = ets.at[1].set(model_output)
            state_dict["ets"] = ets

            state_dict["model_output"] = (3 * ets[1] - ets[0]) / 2
            state_dict["sample"] = sample
            return state_dict

        def counter_3(state_dict):
            ets = state_dict["ets"]
            ets = ets.at[2].set(model_output)
            state_dict["ets"] = ets

            state_dict["model_output"] = (23 * ets[2] - 16 * ets[1] + 5 * ets[0]) / 12
            state_dict["sample"] = sample
            return state_dict

        def counter_other(state_dict):
            ets = state_dict["ets"]
            ets = ets.at[3].set(model_output)

            state_dict["model_output"] = (1 / 24) * (55 * ets[3] - 59 * ets[2] + 37 * ets[1] - 9 * ets[0])
            state_dict["sample"] = sample

            ets = ets.at[0].set(ets[1])
            ets = ets.at[1].set(ets[2])
            ets = ets.at[2].set(ets[3])
            state_dict["ets"] = ets

            return state_dict

        counter = jnp.clip(state.counter, 0, 4)
        state_dict = jax.lax.switch(
            counter,
            [counter_0, counter_1, counter_2, counter_3, counter_other],
            state_dict,
        )

        sample = state_dict["sample"]
        model_output = state_dict["model_output"]
        prev_sample = self._get_prev_sample(state_dict, sample, timestep, prev_timestep, model_output)
        state_dict["counter"] += 1

        return {"prev_sample": prev_sample}, state_dict

    def _get_prev_sample(
        self,
        state_dict: dict,
        sample,
        timestep,
        timestep_prev,
        model_output
    ):
        # See formula (9) of PNDM paper https://arxiv.org/pdf/2202.09778.pdf
        # this function computes x_(t−δ) using the formula of (9)
        # Note that x_t needs to be added to both sides of the equation

        # Notation (<variable name> -> <name in paper>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        state = PNDMSchedulerState.from_state_dict(state_dict)
        alpha_prod_t = state.alphas_cumprod[timestep + 1 - state._offset]
        alpha_prod_t_prev = state.alphas_cumprod[timestep_prev + 1 - state._offset]
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    def __len__(self):
        return self.config.num_train_timesteps
