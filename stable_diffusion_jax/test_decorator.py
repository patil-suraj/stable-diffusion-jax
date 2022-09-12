import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import flax.linen as nn
import jax.numpy as jnp
from jax import random, tree_map
import jax
import functools
import inspect
import functools
from diffusers.configuration_utils import ConfigMixin

def register_to_config(cls):
    original_init = cls.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        # Ignore private kwargs in the init.
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        # original_init(self, *args, **init_kwargs)
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )

        ignore = getattr(self, "ignore_for_config", [])
        # Get positional arguments aligned with kwargs
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default for i, (name, p) in enumerate(signature.parameters.items()) if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg

        # Then add all kwargs
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        getattr(self, "register_to_config")(**new_kwargs)
        original_init(self, *args, **init_kwargs)
    
    cls.__init__ = init
    return cls

class ModelMixin():
    # config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    testatr = "haha"

    def smile(self):
        print("haha")

@register_to_config
class MLP(nn.Module, ModelMixin, ConfigMixin):
    n1: int
    dtype: jnp.dtype = jnp.bfloat16
    config_name: str = "config.json"

    def setup(self):
        self.dense1 = nn.Dense(self.n1)

    def __call__(self, x):
        x = self.dense1(x)
        x = nn.relu(x)
        return x

model = MLP(n1=2)
# print(model)
params = model.init(random.PRNGKey(0), random.normal(random.PRNGKey(1), (1,1)))
# print(tree_map(lambda x: x.shape, params))
print(model.testatr)