from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Int, PRNGKeyArray
from kira.model.mha import MultiheadAttention
from kira.model.rope_embeddings import RotaryPositionalEmbedding
from functools import partial



class Block(eqx.nn.StatefulLayer):
    mha_attention: MultiheadAttention
    rms_norm: eqx.nn.RMSNorm
    feedforward: eqx.nn.MLP
    dropout: eqx.nn.Dropout

    n_embd: int = eqx.field(static=True)
    max_seq_len: int = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    num_query_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)

    key_rope_embeddings: RotaryPositionalEmbedding
    query_rope_embeddings: RotaryPositionalEmbedding

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
        self.query_rope_embeddings = RotaryPositionalEmbedding(
            embedding_size=n_embd, max_seq_len=max_seq_len
        )

        self.key_rope_embeddings = RotaryPositionalEmbedding(
            embedding_size=n_embd, max_seq_len=max_seq_len
        )

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

        self.rms_norm = eqx.nn.RMSNorm(shape=n_embd)

        self.feedforward = eqx.nn.MLP(
            n_embd, out_size=n_embd, width_size=width_size, depth=depth, key=subkeys[1]
        )

        self.dropout = eqx.nn.Dropout(p=p)

    def __call__(
        self,
        x: Int[Array, "max_seq_len input_dim"],
        state: Optional[eqx.nn.State] = None,
        mask: Optional[str] = "causal",
        *,
        key: Optional[PRNGKeyArray],
        **kwargs,
    ):
        def process_heads(query_heads, key_heads, value_heads):
            query_heads = jax.vmap(self.query_rope_embeddings, in_axes=1, out_axes=1)(
                query_heads
            )
            key_heads = jax.vmap(self.key_rope_embeddings, in_axes=1, out_axes=1)(
                key_heads
            )

            return query_heads, key_heads, value_heads

        mha_partial = partial(
            self.mha_attention,
            process_heads=process_heads,
            query=jax.vmap(self.rms_norm)(x),
            key_=jax.vmap(self.rms_norm)(x),
            value=jax.vmap(self.rms_norm)(x),
            mask=mask,
        )
        if state is not None:
            mha, state = mha_partial(state=state, key=key)
        else:
            mha = mha_partial(key=key)
        x = mha + x
        inference = True if key is None else False
        d_key1 = None
        d_key2 = None
        if not inference and key is not None:
            key, d_key1, d_key2 = jax.random.split(key, 3)
        x = self.dropout(x, key=d_key1, inference=inference)
        ff = jax.vmap(self.feedforward)(jax.vmap(self.rms_norm)(x))
        x = ff + x
        x = self.dropout(x, key=d_key2, inference=inference)
        return x, state


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

    rms_norm: eqx.nn.RMSNorm

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
        self.rms_norm = eqx.nn.RMSNorm(shape=n_embd)
        self.output = eqx.nn.Linear(n_embd, n_dims, key=subkeys[-1])

    def __call__(
        self,
        x: Int[Array, "seq_len"],
        state: Optional[eqx.nn.State] = None,
        mask: Optional[str] = "causal",
        *,
        key: Optional[PRNGKeyArray],
    ):
        x = jax.vmap(self.input_embedding)(x)
        x, state = self.blocks(x, state, mask, key=key)
        x = jax.vmap(self.rms_norm)(x)
        x = jax.vmap(self.output)(x)

        if state is not None:
            return x, state
        else:
            return x
