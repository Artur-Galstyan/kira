import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float
from typing import Union
import math
from dataclasses import dataclass

from numpy import who


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


def discretize(A: Array, B: Array, C: Array, step):
    N, _ = A.shape
    I = jnp.eye(N)
    inv = jnp.linalg.inv(I - (step / 2) * A)
    B_ = (inv * step) @ B
    A_ = inv @ (I + (step / 2) * A)

    return A_, B_, C


class Mamba(eqx.Module):
    model_args: ModelArgs = eqx.field(static=True)

    embedding: eqx.nn.Embedding
    layers: eqx.nn.Sequential

    def __init__(self, model_args: ModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        key, *subkeys = jax.random.split(key, 1 + model_args.n_layer)
        self.embedding = eqx.nn.Embedding(
            model_args.vocab_size, model_args.d_model, key=subkeys[0]
        )


class ResidualBlock(eqx.Module):
    pass


class MambaBlock(eqx.Module):
    d_model: int = eqx.field(static=True)
    d_inner: int = eqx.field(static=True)
    d_conv: int = eqx.field(static=True)
    dt_rank: int = eqx.field(static=True)
    d_state: int = eqx.field(static=True)

    in_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d

    x_proj: eqx.nn.Linear
    dt_proj: eqx.nn.Linear

    A_log: Array
    D: Array

    out_proj: eqx.nn.Linear

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        d_conv: int,
        dt_rank: int,
        d_state: int,
        use_linear_bias: bool,
        use_conv_bias: bool,
        *,
        key: PRNGKeyArray,
    ):
        (
            key,
            linear_key,
            conv1d_key,
            x_proj_key,
            dt_proj_key,
            out_proj_key,
        ) = jax.random.split(key, 6)
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.dt_rank = dt_rank
        self.d_state = d_state

        self.in_proj = eqx.nn.Linear(
            d_model,
            d_inner * 2,
            use_bias=use_linear_bias,
            key=linear_key,
        )

        self.conv1d = eqx.nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            use_bias=use_conv_bias,
            groups=d_inner,
            padding=d_conv - 1,
            key=conv1d_key,
        )

        self.x_proj = eqx.nn.Linear(
            d_inner, dt_rank + d_state * 2, use_bias=False, key=x_proj_key
        )

        self.dt_proj = eqx.nn.Linear(dt_rank, d_inner, use_bias=True, key=dt_proj_key)

        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner).reshape(d_inner, d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner)
        self.out_proj = eqx.nn.Linear(
            d_inner, d_model, use_bias=use_linear_bias, key=x_proj_key
        )

    def __call__(self, x: Array):
        seq_len, d = x.shape
        x_and_res = jax.vmap(self.in_proj)(x)
        (x, res) = jnp.split(x_and_res, 2, axis=-1)

        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_len]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

        y = self.ssm(x)

        y = y * jax.nn.silu(res)

        output = jax.vmap(self.out_proj)(y)

        return output

    def ssm(self, x: Float[Array, "seq_len d_inner"]):
        d_in, n = self.A_log.shape

        A = -jnp.exp(self.A_log)
        D = self.D

        x_dbl = jax.vmap(self.x_proj)(x)
        delta, B, C = jnp.split(x_dbl, 3, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.dt_proj)(delta))

        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        L, d_in = u.shape
        n = A.shape[1]

        delta_A = jnp.exp(jnp.einsum("l d,d n -> l d n", delta, A))
        delta_B_u = jnp.einsum("l d,l n,l d -> l d n", delta, B, u)

        x = jnp.zeros(shape=(d_in, n))

        def step(x, i):
            x = delta_A[i] * x + delta_B_u[i]
            y = jnp.einsum("d n,n ->d", x, C[:, i])
            return x, y

        _, ys = jax.lax.scan(step, x, jnp.arange(L))

        ys = ys + u * D
        return ys


class Mamba(eqx.Module):
    pass


if __name__ == "__main__":
    block = MambaBlock(
        d_model=512,
        d_inner=1024,
        d_conv=4,
        dt_rank=3,
        d_state=3,
        use_linear_bias=True,
        use_conv_bias=True,
        key=jax.random.PRNGKey(0),
    )
    x = jnp.ones((3, 512))
    block(x)
    print("done")
