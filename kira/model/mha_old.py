import functools as ft
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from kira.model.rope_embeddings import RotaryPositionalEmbedding


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


def vmapped_attention(
    query_heads,
    key_heads,
    value_heads,
    dropout=None,
    inference=None,
    mask=None,
    keys=None,
):
    attn_fn = ft.partial(
        dot_product_attention, dropout=dropout, inference=inference, key=keys, mask=mask
    )
    # Batch `keys` down its first axis as it is passed as a keyword argument.
    dpa = jax.vmap(
        lambda q, k, v: attn_fn(q, k, v),
        in_axes=(1, None, None),
        out_axes=1,
    )(query_heads, key_heads, value_heads)
    return dpa


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

    max_seq_len: int = eqx.field(static=True)

    key_rope_embeddings: RotaryPositionalEmbedding
    query_rope_embeddings: RotaryPositionalEmbedding

    def __init__(
        self,
        query_embedding_dim: int,
        key_embedding_dim: int,
        value_embedding_dim: int,
        query_input_dim: int,
        key_input_dim: int,
        value_input_dim: int,
        output_dim: int,
        num_heads: int,
        query_multihead_dim: int,
        kv_multihead_dim: int,
        key: PRNGKeyArray,
        max_seq_len: int = None,
    ):
        key, qkey, kkey, vkey, okey = jax.random.split(key, 5)
        assert (
            query_multihead_dim == num_heads
        ), "query_multihead_dim must be equal to num_heads"
        self.query_projection = eqx.nn.Linear(
            query_input_dim,
            query_multihead_dim * query_embedding_dim,
            key=qkey,
            use_bias=False,
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
        query = jax.vmap(self.query_projection)(x).reshape(
            self.max_seq_len, self.query_multihead_dim, self.query_embedding_dim
        )
        query = jax.vmap(self.query_rope_embeddings, in_axes=1, out_axes=1)(query)
        key_ = jax.vmap(self.key_projection)(x).reshape(
            self.max_seq_len, self.kv_multihead_dim, self.key_embedding_dim
        )
        key_ = jax.vmap(self.key_rope_embeddings, in_axes=1, out_axes=1)(key_)

        value = jax.vmap(self.value_projection)(x).reshape(
            self.max_seq_len, self.kv_multihead_dim, self.value_embedding_dim
        )

        mask = jnp.tril(jnp.ones(shape=(self.max_seq_len, self.max_seq_len))) == 1

        if (
            self.kv_multihead_dim == self.num_heads
            and self.query_multihead_dim == self.num_heads
        ):
            dpa = ft.partial(
                dot_product_attention,
                mask=mask,
            )
            attn = jax.vmap(dpa, in_axes=1, out_axes=1)(query, key_, value)
        else:
            # interpolate MQA and MHA
            pt_vmapped_attention = ft.partial(
                vmapped_attention,
                mask=mask,
            )
            attn = jax.vmap(pt_vmapped_attention, in_axes=(None, 1, 1), out_axes=1)(
                query, key_, value
            )
            attn = jnp.sum(attn, axis=1)
            # Taking the mean over the d dimension
            attn = attn / self.kv_multihead_dim

        concatenation = attn.reshape(self.max_seq_len, -1)
        output = jax.vmap(self.output)(concatenation)

        return output
