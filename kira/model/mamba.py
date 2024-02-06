import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Array, Float, Int
from typing import Union
import math
from dataclasses import dataclass

from kira.model.rope_embeddings import RotaryPositionalEmbedding


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
    max_seq_len: int = 8

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba(eqx.Module):
    model_args: ModelArgs = eqx.field(static=True)
    layers: eqx.nn.Sequential
    norm_f: eqx.nn.RMSNorm
    shared_emb_lm_head: eqx.nn.Shared

    def __init__(self, model_args: ModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        key, *subkeys = jax.random.split(key, 1 + model_args.n_layer)
        embedding = eqx.nn.Embedding(
            model_args.vocab_size, model_args.d_model, key=subkeys[0]
        )

        self.layers = eqx.nn.Sequential(
            [
                ResidualBlock(model_args, key=subkeys[i])
                for i in range(model_args.n_layer)
            ],
        )

        self.norm_f = eqx.nn.RMSNorm(model_args.d_model)
        lm_head = eqx.nn.Linear(
            model_args.d_model,
            model_args.vocab_size,
            use_bias=False,
            key=subkeys[-1],
        )
        where = lambda embed_and_lin: embed_and_lin[1].weight  # noqa
        get = lambda embed_and_lin: embed_and_lin[0].weight  # noqa
        self.shared_emb_lm_head = eqx.nn.Shared(
            (embedding, lm_head), where=where, get=get
        )

    def __call__(
        self,
        x: Int[Array, "seq_len"],
        *,
        state: eqx.nn.State = None,
        key: PRNGKeyArray = None,
    ) -> Float[Array, "seq_len vocab_size"]:
        embedding, linear = self.shared_emb_lm_head()
        x = jax.vmap(embedding)(x)

        x = self.layers(x)
        x = jax.vmap(self.norm_f)(x)
        logits = jax.vmap(linear)(x)
        return logits


class MambaBlock(eqx.Module):
    model_args: ModelArgs = eqx.field(static=True)

    in_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d

    x_proj: eqx.nn.Linear
    dt_proj: eqx.nn.Linear

    A_log: Array
    D: Array

    out_proj: eqx.nn.Linear

    def __init__(
        self,
        model_args: ModelArgs,
        *,
        key: PRNGKeyArray,
    ):
        self.model_args = model_args
        (
            key,
            linear_key,
            conv1d_key,
            x_proj_key,
            dt_proj_key,
            out_proj_key,
        ) = jax.random.split(key, 6)

        self.in_proj = eqx.nn.Linear(
            model_args.d_model,
            model_args.d_inner * 2,
            use_bias=model_args.bias,
            key=linear_key,
        )

        self.conv1d = eqx.nn.Conv1d(
            in_channels=model_args.d_inner,
            out_channels=model_args.d_inner,
            kernel_size=model_args.d_conv,
            use_bias=model_args.conv_bias,
            groups=model_args.d_inner,
            padding=model_args.d_conv - 1,
            key=conv1d_key,
        )

        self.x_proj = eqx.nn.Linear(
            model_args.d_inner,
            model_args.dt_rank + model_args.d_state * 2,
            use_bias=False,
            key=x_proj_key,
        )

        self.dt_proj = eqx.nn.Linear(
            model_args.dt_rank, model_args.d_inner, use_bias=True, key=dt_proj_key
        )

        A = jnp.repeat(
            jnp.arange(1, model_args.d_state + 1), model_args.d_inner
        ).reshape(model_args.d_inner, model_args.d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(model_args.d_inner)
        self.out_proj = eqx.nn.Linear(
            model_args.d_inner,
            model_args.d_model,
            use_bias=model_args.bias,
            key=x_proj_key,
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
        split_indices = [self.model_args.dt_rank, self.model_args.dt_rank + n]
        delta, B, C = jnp.split(x_dbl, split_indices, axis=-1)
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

            y = jnp.einsum("d n,n -> d", x, C[i, :])
            return x, y

        _, ys = jax.lax.scan(step, x, jnp.arange(L))

        ys = ys + u * D
        return ys


class ResidualBlock(eqx.Module):
    mamba_block: MambaBlock
    model_args: ModelArgs = eqx.field(static=True)
    rns_norm: eqx.nn.RMSNorm

    def __init__(self, model_args: ModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        self.mamba_block = MambaBlock(
            model_args=model_args,
            key=key,
        )
        self.rns_norm = eqx.nn.RMSNorm(model_args.d_model)

    def __call__(self, x: Array, *, key: PRNGKeyArray = None) -> Array:
        return self.mamba_block(jax.vmap(self.rns_norm)(x)) + x


if __name__ == "__main__":
    model_args = ModelArgs(
        d_model=512,
        n_layer=6,
        vocab_size=256,
    )
    key = jax.random.PRNGKey(0)
    mamba = Mamba(model_args, key=key)
    x = jnp.ones((8), dtype=jnp.int32)
    print(mamba(x).shape)
