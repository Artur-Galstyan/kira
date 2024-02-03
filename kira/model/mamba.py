import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import jaxtyping as jt


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

    A_log: jt.Array
    D: jt.Array

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

        A = jnp.repeat(jnp.arange(1, d_state + 1), d_inner)
        self.A_log = jnp.log(A)
        self.D = jnp.ones(d_inner)
        self.out_proj = eqx.nn.Linear(
            d_inner, d_model, use_bias=use_linear_bias, key=x_proj_key
        )

    def __call__(self, x: jt.Array):
        seq_len, d = x.shape

        x_and_res = jax.vmap(self.in_proj)(x)

        (x, res) = jnp.split(x_and_res, 2, axis=-1)

        x = jnp.transpose(x)
        x = self.conv1d(x)[:, :seq_len]
        x = jnp.transpose(x)
        x = jax.nn.silu(x)

    def ssm(self, x: jt.Array):
        d_in, n = self.A_log.shape


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
