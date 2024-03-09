import equinox as eqx
import jax
import jax.numpy as jnp
from kira.model_args import LLaMAModelArgs
from kira.models.llama.llama2 import LLaMA


def main():
    llama_args = LLaMAModelArgs(dim=128, n_layers=4, n_heads=4, vocab_size=384)

    llama, state = eqx.nn.make_with_state(LLaMA)(
        model_args=llama_args, key=jax.random.PRNGKey(0)
    )

    test_x = jnp.ones((8,), dtype=jnp.int32)
    y, state = llama(test_x, state, key=None, inference=True)
    print(y.shape)


if __name__ == "__main__":
    main()
