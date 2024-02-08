from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Complex, Float, PRNGKeyArray


class RotaryPositionalEmbedding(eqx.Module):
    embedding_size: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    freqs_cis: Complex[Array, "embedding_size/2"] = eqx.field(static=True)

    def __init__(
        self,
        embedding_size: int,
        max_seq_len: int,
        *,
        key: Optional[PRNGKeyArray] = None,
        **kwargs,
    ):
        """**Arguments:**
        `RotaryPositionalEmbedding` requires:
        - `embedding_size`: Size of the token embeddings. Must be non-negative.
        - `max_seq_len`: The maximum sequence length. Must be non-negative.
        - `key`: Not used; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        """
        super().__init__(**kwargs)
        if embedding_size < 0:
            raise ValueError("embedding_size must not be negative.")
        if max_seq_len < 0:
            raise ValueError("max_seq_len must not be negative.")
        self.embedding_size = embedding_size
        self.max_seq_len = max_seq_len
        self.freqs_cis = self.precompute_freqs_cis(embedding_size, max_seq_len)

    @staticmethod
    def negate_half(x: Float[Array, "max_seq_len embedding_size"]):
        d_2 = x.shape[-1] // 2
        return jnp.concatenate([x[..., :d_2], -x[..., d_2:]], axis=-1)

    @staticmethod
    def precompute_freqs_cis(
        embedding_size: int, end: int, theta: float = 10000.0
    ) -> Complex[Array, "end embedding_size/2"]:
        def polar(abs, angle):
            return jnp.array(
                abs * jnp.cos(angle) + abs * jnp.sin(angle) * 1j, dtype=jnp.complex64
            )

        freqs = 1.0 / (
            theta ** (jnp.arange(0, embedding_size, 2)[jnp.newaxis, :] / embedding_size)
        )
        t = jnp.arange(end)
        freqs_outer = jnp.outer(t, freqs)
        freqs_cis = polar(jnp.ones_like(freqs_outer), freqs_outer)
        return jax.lax.stop_gradient(freqs_cis)

    def __call__(
        self,
        x: Float[Array, "max_seq_len embedding_size"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "max_seq_len embedding_size"]:
        """**Arguments:**
        - `x`: A JAX array of shape `(max_seq_len, embedding_size)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)
        **Returns:**
        A JAX array of shape `(max_seq_len, embedding_size)`, with the rotary positional
        encoding applied to the input.
        """
        assert x.ndim == 2, f"x.ndim must be 2, but {x.ndim} != 2."
        seq_len, embedding_size = x.shape
        assert embedding_size == self.embedding_size, (
            f"x.shape[-1] must match self.embedding_size, "
            f"but {x.shape[-1]} != {self.embedding_size}"
        )
        assert (
            embedding_size % 2 == 0
        ), f"x.shape[-1] must be even, but {x.shape[-1]} is not even."
        assert seq_len <= self.max_seq_len, (
            f"x.shape[0] must be <= self.max_seq_len, "
            f"but {x.shape[0]} > {self.max_seq_len}"
        )
        neg_half_x = self.negate_half(x)
        freqs_real = jnp.tile(self.freqs_cis.real, (1, 2))
        freqs_imag = jnp.tile(self.freqs_cis.imag, (1, 2))
        x_rope = (x * freqs_real) + (neg_half_x * freqs_imag)
        return x_rope
