import jax
import jax.numpy as jnp

import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # argh honeydew cuda dependency issues :( let's just use cpu for now
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BATCH = 1
HEADS = 4 
SEQUENCE = 2048
HEAD_DIM = 128

# init Q, K, V with randoms instead of FFN for now
Q = jax.random.normal( jax.random.key(0), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
K = jax.random.normal( jax.random.key(1), (BATCH, SEQUENCE, HEADS, HEAD_DIM))
V = jax.random.normal( jax.random.key(2), (BATCH, SEQUENCE, HEADS, HEAD_DIM))

def my_attention(_Q, _K, _V):
  scale = 1.0 / jnp.sqrt(HEAD_DIM)
  
  _W_unnormalized = jnp.einsum('BSHD,BTHD->BHST', _Q, _K) * scale # dot product along vectors along HEAD_DIM dimension
  _W = jax.nn.softmax(_W_unnormalized, axis=-1) # softmax(Q@K.T)
  out = jax.numpy.einsum('BHST,BTHD->BSHD', _W, _V) # softmax(Q@K.T) @ V err einsum is confusing
  return out

my_attn_output = my_attention(Q, K, V)
real_attn_output = jax.nn.dot_product_attention(Q,K,V, is_causal=False)

assert jnp.allclose(my_attn_output, real_attn_output, rtol=1e-1, atol=1e-1), "not close"

  