from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray
from kira.model.mha import MultiheadAttention


class RMSNorm(eqx.Module):
    weight: Array
    eps: float

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = jnp.ones(dim)

    def _norm(self, x: Array):
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, x: Array) -> Array:
        output = self._norm(x)
        return output * self.weight


class Block(eqx.Module):
    mha_attention: MultiheadAttention
    rms_norm: RMSNorm
    feedforward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    n_embd: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    num_query_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        num_query_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        width_size: int = 64,
        depth: int = 1,
        p: float = 0.2,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        key, *subkeys = jax.random.split(key, 5)
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.mha_attention = MultiheadAttention(
            num_heads=num_heads,
            query_size=n_embd,
            qk_size=n_embd,
            key_size=n_embd,
            value_size=n_embd,
            vo_size=n_embd,
            output_size=n_embd,
            query_multihead_dim=num_query_heads,
            kv_multihead_dim=num_kv_heads,
            state_length=max_seq_len,
            key=subkeys[0],
        )

        self.rms_norm = RMSNorm(dim=n_embd)

        self.feedforward = eqx.nn.MLP(
            n_embd, out_size=n_embd, width_size=width_size, depth=depth, key=subkeys[1]
        )

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(
        self,
        x: Int[Array, "max_seq_len input_dim"],
        *,
        key: Optional[PRNGKeyArray],
        **kwargs,
    ):
        mha = self.mha_attention(
            query=self.rms_norm(x),
            key_=self.rms_norm(x),
            value=self.rms_norm(x),
        )
        x = mha + x
        inference = True if key is None else False
        d_key1 = None
        d_key2 = None
        if not inference and key is not None:
            key, d_key1, d_key2 = jax.random.split(key, 3)
        x = self.dropout(x, key=d_key1, inference=inference)
        ff = jax.vmap(self.feedforward)(self.rms_norm(x))
        x = ff + x
        x = self.dropout(x, key=d_key2, inference=inference)
        return x


class Kira(eqx.Module):
    input_embedding: eqx.nn.Embedding

    n_dims: int = eqx.field(static=True)
    n_embd: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    num_query_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)

    blocks: eqx.nn.Sequential

    output: eqx.nn.Linear

    rms_norm: RMSNorm

    def __init__(
        self,
        n_dims: int,
        n_embd: int,
        num_heads: int,
        num_query_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        n_layers: int = 4,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads

        key, *subkeys = jax.random.split(key, n_layers + 2)

        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])
        self.blocks = eqx.nn.Sequential(
            [
                Block(
                    n_embd,
                    num_heads,
                    num_query_heads,
                    num_kv_heads,
                    max_seq_len,
                    key=subkeys[i + 1],
                )
                for i in range(n_layers)
            ]
        )
        self.rms_norm = RMSNorm(dim=n_embd)
        self.output = eqx.nn.Linear(n_embd, n_dims, key=subkeys[-1])

    def __call__(
        self,
        x: Int[Array, "seq_len"],
        key: Optional[PRNGKeyArray],
    ):
        x = jax.vmap(self.input_embedding)(x)
        x = self.blocks(x, key=key)
        x = self.rms_norm(x)
        x = jax.vmap(self.output)(x)

        return x
