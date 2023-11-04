import functools as ft

import equinox as eqx
import jax.numpy as jnp
from icecream import ic
from jaxtyping import Array


class Counter(eqx.nn.StatefulLayer):
    index: eqx.nn.StateIndex

    def __init__(self):
        init_state = 0
        self.index = eqx.nn.StateIndex(init_state)

    def __call__(
            self, x: Array, state: eqx.nn.State, *, key
    ) -> tuple[Array, eqx.nn.State]:
        value = state.get(self.index)
        ic(x.shape, value.shape)
        new_x = x + value
        new_state = state.set(self.index, value + 1)
        ic(new_state)
        return new_x, new_state


class OuterModel(eqx.Module):
    counters: eqx.nn.Sequential

    def __init__(self):
        self.counters = eqx.nn.Sequential([Counter() for _ in range(3)])

    def __call__(
            self, x: Array, state: eqx.nn.State, *, key, **kwargs
    ) -> tuple[Array, eqx.nn.State]:
        x, state = self.counters(x, state)

        return x, state


def main():
    model, state = eqx.nn.make_with_state(OuterModel)()
    ic(state)
    x = jnp.ones(shape=(2, 3))

    partial_model = ft.partial(model, state=state, key=None)
    output, state = eqx.filter_vmap(partial_model)(x)
    ic(output, state)

    partial_model = ft.partial(model, state=state, key=None)
    output, state = eqx.filter_vmap(partial_model)(x)
    ic(state)


if __name__ == "__main__":
    main()
