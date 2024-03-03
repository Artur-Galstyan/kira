from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from kira.model_args import MambaModelArgs


def selective_scan(
    x: Float[Array, "seq_length d_inner"],
    delta: Float[Array, "seq_length d_inner"],
    A: Float[Array, "d_inner d_state"],
    B: Float[Array, "seq_length d_state"],
    C: Float[Array, "seq_length d_state"],
    D: Float[Array, " d_inner"],
) -> Float[Array, "seq_length d_inner"]:
    L, d_inner = x.shape
    _, d_state = A.shape
    delta_A = jnp.exp(jnp.einsum("l d,d n -> l d n", delta, A))
    delta_B_u = jnp.einsum("l d,l n,l d -> l d n", delta, B, x)

    x_res = jnp.zeros(shape=(d_inner, d_state))

    def step(x, i):
        x = delta_A[i] * x + delta_B_u[i]

        y = jnp.einsum("d n,n -> d", x, C[i, :])
        return x, y

    _, ys = jax.lax.scan(step, x_res, jnp.arange(L))

    ys = ys + x * D
    return ys


class Mamba(eqx.Module):
    model_args: MambaModelArgs = eqx.field(static=True)
    layers: eqx.nn.Sequential
    norm_f: eqx.nn.RMSNorm
    shared_emb_lm_head: eqx.nn.Shared

    def __init__(self, model_args: MambaModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        if model_args.n_layers is None:
            raise ValueError("n_layers must be provided")
        key, *subkeys = jax.random.split(key, 1 + model_args.n_layers)
        embedding = eqx.nn.Embedding(
            model_args.n_dims, model_args.n_embd, key=subkeys[0]
        )

        self.layers = eqx.nn.Sequential(
            [
                ResidualBlock(model_args, key=subkeys[i])
                for i in range(model_args.n_layers)
            ],
        )

        self.norm_f = eqx.nn.RMSNorm(model_args.n_embd)
        lm_head = eqx.nn.Linear(
            model_args.n_embd,
            model_args.n_dims,
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
        x: Int[Array, " seq_length"],
        *,
        state: Optional[eqx.nn.State] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_length n_dims"]:
        embedding, linear = self.shared_emb_lm_head()
        x = jax.vmap(embedding)(x)

        x = self.layers(x)
        x = jax.vmap(self.norm_f)(x)
        logits = jax.vmap(linear)(x)
        return logits


class SSM(eqx.Module):
    input_proj: eqx.nn.Linear
    delta_proj: eqx.nn.Linear
    A_log: Float[Array, "d_inner d_state"]
    D: Float[Array, " d_inner"]

    model_args: MambaModelArgs = eqx.field(static=True)

    def __init__(self, model_args: MambaModelArgs, *, key: PRNGKeyArray):
        if model_args.d_inner is None:
            raise ValueError("d_inner must be provided")
        if model_args.dt_rank is None:
            raise ValueError("dt_rank must be provided")
        if not isinstance(model_args.dt_rank, int):
            raise ValueError("dt_rank must be an integer")
        (
            key,
            x_proj_key,
            dt_proj_key,
        ) = jax.random.split(key, 3)
        self.model_args = model_args
        self.input_proj = eqx.nn.Linear(
            model_args.d_inner,
            model_args.dt_rank + model_args.d_state * 2,
            use_bias=False,
            key=x_proj_key,
        )

        self.delta_proj = eqx.nn.Linear(
            model_args.dt_rank, model_args.d_inner, use_bias=True, key=dt_proj_key
        )

        A = jnp.repeat(
            jnp.arange(1, model_args.d_state + 1), model_args.d_inner
        ).reshape(model_args.d_inner, model_args.d_state)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(model_args.d_inner)

    def __call__(self, x: Float[Array, "seq_length d_inner"]):
        assert isinstance(self.model_args.dt_rank, int), "dt_rank must be an integer"
        A = -jnp.exp(self.A_log)
        D = self.D

        x_dbl = jax.vmap(self.input_proj)(x)

        split_indices = [
            self.model_args.dt_rank,
            self.model_args.dt_rank + self.model_args.d_state,
        ]
        delta, B, C = jnp.split(x_dbl, split_indices, axis=-1)
        delta = jax.nn.softplus(jax.vmap(self.delta_proj)(delta))

        y = selective_scan(x, delta, A, B, C, D)
        return y


class MambaBlock(eqx.Module):
    model_args: MambaModelArgs = eqx.field(static=True)

    in_proj: eqx.nn.Linear
    conv1d: eqx.nn.Conv1d
    ssm: SSM
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        model_args: MambaModelArgs,
        *,
        key: PRNGKeyArray,
    ):
        assert model_args.d_inner is not None, "d_inner must be provided"
        assert isinstance(model_args.dt_rank, int), "dt_rank must be an integer"
        self.model_args = model_args

        (
            key,
            linear_key,
            conv1d_key,
            ssm_key,
            out_proj_key,
        ) = jax.random.split(key, 5)

        self.in_proj = eqx.nn.Linear(
            model_args.n_embd,
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
        self.ssm = SSM(model_args, key=ssm_key)
        self.out_proj = eqx.nn.Linear(
            model_args.d_inner,
            model_args.n_embd,
            use_bias=model_args.bias,
            key=out_proj_key,
        )

    def __call__(self, x: Array):
        seq_length, d = x.shape
        x_and_res = jax.vmap(self.in_proj)(x)

        (x, res) = jnp.split(x_and_res, 2, axis=-1)
        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_length]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

        y = self.ssm(x)
        y = y * jax.nn.silu(res)

        output = jax.vmap(self.out_proj)(y)
        return output


class ResidualBlock(eqx.Module):
    mamba_block: MambaBlock
    model_args: MambaModelArgs = eqx.field(static=True)
    rns_norm: eqx.nn.RMSNorm

    def __init__(self, model_args: MambaModelArgs, *, key: PRNGKeyArray):
        self.model_args = model_args
        self.mamba_block = MambaBlock(
            model_args=model_args,
            key=key,
        )
        self.rns_norm = eqx.nn.RMSNorm(model_args.n_embd)

    def __call__(
        self,
        x: Float[Array, "seq_length n_embd"],
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Array:
        return self.mamba_block(jax.vmap(self.rns_norm)(x)) + x


if __name__ == "__main__":
    model_args = MambaModelArgs(
        n_embd=512,
        n_layers=6,
        n_dims=256,
    )
    key = jax.random.PRNGKey(model_args.key_seed)
    mamba = Mamba(model_args, key=key)
    x = jnp.ones((8), dtype=jnp.int32)
    print(mamba(x).shape)
