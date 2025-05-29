import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state # jax's immutatble state philosophy
                                      # in pytorch, this would be the model, optimizer, scheduler, etc
import flax.linen.attention as attention

import numpy as np
import optax

# HYPERPARAMETERS
BATCH_IN_SEQUENCES = 256
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

class OurModel(nn.Module):
  @nn.compact # alternative to setup() approach
  def __call__(self, input_tokens):
    '''
    input_tokens is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
      'embedding',                # param name
      nn.initializers.normal(1),  # initializer func
      (VOCAB_DIM, EMBED_DIM),     # shape
      jnp.float32,                # dtype
    )

    x = jnp.asarray(embedding)[input_tokens] #BATCH, SEQUENCE, EMBED

    pos_embedding = self.param(
      'pos_embedding',
      nn.initializers.normal(1),
      (1, SEQUENCE_LENGTH, EMBED_DIM),
      jnp.float32,
    )

    x += jnp.asarray(pos_embedding) # BATCH, SEQUENCE, EMBED

    for i in range(LAYERS):
      layer_input = x
      emb2ff = self.param(
        'emb2ff_' + str(i),
        nn.initalizers.lecun_normal(),
        (EMBED_DIM, FF_DIM),
        jnp.float32,
      )

      x = x @ jnp.asarray(emb2ff) # [B,S,E] @ [E, F] = [BATCH, SEQUENCE, FF_DIM]
      x = jax.nn.relu(x)  # why jax.nn.relu instead of flax.linen.relu?
      ff2emb = self.param(
        'ff2emb_' + str(i),
        nn.initializers.lecun_normal(),
        (FF_DIM, EMBED_DIM),
        jnp.float32,
      )
      x = x @ jnp.asarray(ff2emb)
      x = jax.nn.relu(x)

      q_proj = self.param(
        'q_proj_' + str(i),
        nn.initializers.lecun_normal(),
        (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
        jnp.float32,
      )
      q = jnp.einsum("BSE,END->BSND", x, jnp.asarray(q_proj)) # [BSE] @ [END] = [BSND]
      
      k_proj = self.param(
        'k_proj_' + str(i),
        nn.initializers.lecun_normal(),
        (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
        jnp.float32,
      )
      k = jnp.einsum("BSE,END->BSND", x, jnp.asarray(k_proj)) # [BSE] @ [END] = [BSND]
      
      v_proj = self.param(
        'v_proj_' + str(i),
        nn.initializers.lecun_normal(),
        (EMBED_DIM, NUM_HEADS, HEAD_DEPTH),
        jnp.float32,
      )
      v = jnp.einsum("BSE,END->BSND", x, jnp.asarray(v_proj))

      post_attention = attention.dot_product_attention(q, k, v) # this is cheating but let it slide rn
      post_attention = jax.numpy.reshape(post_attention, (BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, NUM_HEADS * HEAD_DEPTH))

      out_proj = self.param(
        'out_proj_' + str(i),
        nn.initializers.lecun_normal(),
        (NUM_HEADS * HEAD_DEPTH, EMBED_DIM),
        jnp.float32,
      )

      x = post_attention @ jnp.asarray(out_proj) # [BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, NUM_HEADS * HEAD_DEPTH] @ [NUM_HEADS * HEAD_DEPTH, EMBED_DIM] = [BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, EMBED_DIM]
      x += layer_input # residual connection

    emb2vocab = self.param(
      'emb2vocab' + str(i),
      nn.initializers.lecun_normal(),
      (EMBED_DIM, VOCAB_DIM),
      jnp.float32,
    )

    x = x @ jnp.asarray(emb2vocab) # [BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, EMBED_DIM] @ [EMBED_DIM, VOCAB_DIM] = [BATCH_IN_SEQUENCES, SEQUENCE_LENGTH, VOCAB_DIM]
    return x



## data loading ##

