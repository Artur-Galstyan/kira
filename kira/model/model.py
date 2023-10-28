import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from kira.model.rope_embeddings import RotaryPositionalEmbedding


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


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query = query / jnp.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    return jax.nn.softmax(logits, axis=-1)  # pyright: ignore


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[eqx.nn.Dropout] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key, inference=inference)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


class MultiheadAttention(eqx.Module):
    query_projection: eqx.nn.Linear
    key_projection: eqx.nn.Linear
    value_projection: eqx.nn.Linear

    query_input_dim: int = eqx.field(static=True)
    query_embedding_dim: int = eqx.field(static=True)

    key_input_dim: int = eqx.field(static=True)
    key_embedding_dim: int = eqx.field(static=True)

    value_input_dim: int = eqx.field(static=True)
    value_embedding_dim: int = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)

    query_multihead_dim: int = eqx.field(static=True)
    kv_multihead_dim: Optional[int | None] = eqx.field(static=True)

    output: eqx.nn.Linear

    query_rope_embeddings: RotaryPositionalEmbedding
    key_rope_embeddings: RotaryPositionalEmbedding

    max_seq_len: int = eqx.field(static=True)

    def __init__(
        self,
        query_embedding_dim: int,
        key_embedding_dim: int,
        value_embedding_dim: int,
        query_input_dim: int,
        key_input_dim: int,
        value_input_dim: int,
        num_heads: int,
        output_dim: int,
        query_multihead_dim: int,
        kv_multihead_dim: int,
        max_seq_len: int,
        key: PRNGKeyArray,
    ):
        key, qkey, kkey, vkey, okey = jax.random.split(key, 5)
        self.query_projection = eqx.nn.Linear(
            query_input_dim, num_heads * query_embedding_dim, key=qkey, use_bias=False
        )
        self.key_projection = eqx.nn.Linear(
            key_input_dim,
            kv_multihead_dim * key_embedding_dim,
            key=kkey,
            use_bias=False,
        )
        self.value_projection = eqx.nn.Linear(
            value_input_dim,
            kv_multihead_dim * value_embedding_dim,
            key=vkey,
            use_bias=False,
        )

        self.output = eqx.nn.Linear(
            num_heads * value_embedding_dim, output_dim, key=okey, use_bias=False
        )

        self.query_rope_embeddings = RotaryPositionalEmbedding(
            embedding_size=query_embedding_dim, max_seq_len=max_seq_len
        )

        self.key_rope_embeddings = RotaryPositionalEmbedding(
            embedding_size=key_embedding_dim, max_seq_len=max_seq_len
        )

        # parameters
        self.query_input_dim = query_input_dim
        self.query_embedding_dim = query_embedding_dim
        self.key_input_dim = key_input_dim
        self.key_embedding_dim = key_embedding_dim
        self.value_input_dim = value_input_dim
        self.value_embedding_dim = value_embedding_dim
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.query_multihead_dim = query_multihead_dim
        self.kv_multihead_dim = kv_multihead_dim
        self.max_seq_len = max_seq_len

    def __call__(self, x: Float[Array, "max_seq_len input_dim"]):
        seq_len, _ = x.shape
        query = jax.vmap(self.query_projection)(x).reshape(
            seq_len, self.num_heads, self.query_embedding_dim
        )
        query = jax.vmap(self.query_rope_embeddings, in_axes=1, out_axes=1)(query)
        key_ = jax.vmap(self.key_projection)(x).reshape(
            seq_len, self.kv_multihead_dim, self.key_embedding_dim
        )
        key_ = jax.vmap(self.key_rope_embeddings, in_axes=1, out_axes=1)(key_)
        value = jax.vmap(self.value_projection)(x).reshape(
            seq_len, self.kv_multihead_dim, self.value_embedding_dim
        )
        T = seq_len
        mask = jnp.tril(jnp.ones(shape=(T, T))) == 1
        dpa = ft.partial(
            dot_product_attention,
            mask=mask,
        )
        attn = jax.vmap(dpa, in_axes=1, out_axes=1)(query, key_, value)

        concatenation = attn.reshape(seq_len, -1)
        output = jax.vmap(self.output)(concatenation)

        return output


class Block(eqx.Module):
    mha_attention: MultiheadAttention
    rms_norm: RMSNorm
    feedforward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    n_embd: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        max_seq_len: int,
        width_size: int = 32,
        depth: int = 2,
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

        self.mha_attention = MultiheadAttention(
            query_input_dim=n_embd,
            query_embedding_dim=n_embd,
            key_input_dim=n_embd,
            key_embedding_dim=n_embd,
            value_input_dim=n_embd,
            value_embedding_dim=n_embd,
            num_heads=num_heads,
            output_dim=n_embd,
            query_multihead_dim=num_heads,
            kv_multihead_dim=num_heads,
            max_seq_len=max_seq_len,
            key=subkeys[0],
        )

        self.rms_norm = RMSNorm(dim=n_embd)

        self.feedforward = eqx.nn.MLP(
            n_embd, out_size=n_embd, width_size=width_size, depth=depth, key=subkeys[1]
        )

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(
        self, x: Int[Array, "max_seq_len"], *, key: Optional[PRNGKeyArray], **kwargs
    ):
        mha = self.mha_attention(x)
        x = self.rms_norm(mha + x)
        inference = True if key is None else False
        d_key1 = None
        d_key2 = None
        if not inference and key is not None:
            key, d_key1, d_key2 = jax.random.split(key, 3)
        x = self.dropout(x, key=d_key1, inference=inference)
        ff = jax.vmap(self.feedforward)(x)
        x = self.rms_norm(ff + x)
        x = self.dropout(x, key=d_key2, inference=inference)
        return x


class Kira(eqx.Module):
    input_embedding: eqx.nn.Embedding

    n_dims: int = eqx.field(static=True)
    n_embd: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

    blocks: eqx.nn.Sequential

    output: eqx.nn.Linear

    rms_norm: RMSNorm

    def __init__(
        self,
        n_dims: int,
        n_embd: int,
        num_heads: int,
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
        key, *subkeys = jax.random.split(key, n_layers + 2)

        self.input_embedding = eqx.nn.Embedding(n_dims, n_embd, key=subkeys[0])

        self.blocks = eqx.nn.Sequential(
            [
                Block(n_embd, num_heads, max_seq_len, key=subkeys[i + 1])
                for i in range(n_layers)
            ]
        )
        self.rms_norm = RMSNorm(dim=n_embd)
        self.output = eqx.nn.Linear(n_embd, n_dims, key=subkeys[-1])

    def __call__(
        self, x: Int[Array, "max_seq_len"], *, key: Optional[PRNGKeyArray], **kwargs
    ):
        x = jax.vmap(self.input_embedding)(x)
        x = self.blocks(x, key=key)
        x = self.rms_norm(x)
        x = jax.vmap(self.output)(x)
        return x
