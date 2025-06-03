import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state # jax's immutatble state philosophy
                                      # in pytorch, this would be the model, optimizer, scheduler, etc
import time
import functools

import numpy as np
import optax

""" suffix key convention from c.ai

Dimension key:

B: batch size
L: sequence length
M: memory length (length of sequence being attended to)
D: model dimension (sometimes called d_model or embedding_dim)
V: vocabulary size
F: feed-forward subnetwork hidden size
H: number of attention heads in a layer
K: size of each attention key or value (sometimes called d_kv)
"""


# HYPERPARAMETERS
BATCH_IN_SEQUENCES = 384
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DIM = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

FSDP = 4
TENSOR = 2

def my_attention(_Q, _K, _V):
  scale = 1.0 / jnp.sqrt(HEAD_DIM)

  _W_unnormalized = jnp.einsum('BSHD,BTHD->BHST', _Q, _K) * scale # dot product along vectors along HEAD_DIM dimension
  _W = jax.nn.softmax(_W_unnormalized, axis=-1) # softmax(Q@K.T)
  out = jax.numpy.einsum('BHST,BTHD->BSHD', _W, _V) # softmax(Q@K.T) @ V err einsum is confusing
  return out #BSHD

class OurModel(nn.Module):
  @nn.compact # alternative to setup() approach
  def __call__(self, x):
    '''
    input_tokens is [BATCH, SEQUENCE]
    '''
    embedding_VD = self.param(
      'embedding',                # param name
      nn.with_partitioning(nn.initializers.normal(1), ("tp", "fsdp")), # TP for axis0; FSDP for axis 1
      (VOCAB_DIM, EMBED_DIM),     # shape, (VD)
      jnp.float32,                # dtype
    )

    x_BLD = embedding_VD[x] #BATCH, SEQUENCE, EMBED (BLD)

    pos_embedding = self.param(
        'pos_embedding',
        nn.with_partitioning(nn.initializers.normal(1), (None, None, "fsdp")),
        (1, SEQUENCE_LENGTH, EMBED_DIM),
        jnp.float32,
    )

    x_BLD += pos_embedding # broadcasting

    for i in range(LAYERS):
      feedforward_DF = self.param(
        'feedforward' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ("fsdp", "tp")),
        (EMBED_DIM, FF_DIM),
        jnp.float32,
      )

      x_BLF = x_BLD @ feedforward_DF # [B,S,E] @ [E, F] = [BATCH, SEQUENCE, FF_DIM]
      x_BLF = jax.nn.relu(x_BLF)  # why jax.nn.relu instead of flax.linen.relu? so many libraries man
      embed_FD = self.param(
        'embed_' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('tp', 'fsdp')),
        (FF_DIM, EMBED_DIM),
        jnp.float32,
      )
      x_BLD = x_BLF @ embed_FD
      x_BLD = jax.nn.relu(x_BLD)

      # get q,k,v with attn
      q_proj_DHK = self.param(
        'qproj_' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
        (EMBED_DIM, NUM_HEADS, HEAD_DIM), # DHK
        jnp.float32,
      )
      q_BLHK = jnp.einsum("BLD,DHK->BLHK", x_BLD, q_proj_DHK)

      k_proj_DHK = self.param(
        'kproj_' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
        (EMBED_DIM, NUM_HEADS, HEAD_DIM), # DHK
        jnp.float32,
      )
      k_BLHK = jnp.einsum("BLD,DHK->BLHK", x_BLD, k_proj_DHK)

      v_proj_DHK = self.param(
        'vproj_' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
        (EMBED_DIM, NUM_HEADS, HEAD_DIM), # DHK
        jnp.float32,
      )
      v_BLHK = jnp.einsum("BLD,DHK->BLHK", x_BLD, v_proj_DHK)
      
      o_BLHD = my_attention(q_BLHK, k_BLHK, v_BLHK) #B,L,H,D
      
      o_proj_HKD = self.param(
        'oproj_' + str(i),
        nn.with_partitioning(nn.initializers.lecun_normal(), ('fsdp', 'tp')),
        (NUM_HEADS, HEAD_DIM, EMBED_DIM), # HKD
        jnp.float32,
      )
      x = jnp.einsum("BSHD,HDE->BSE", o_BLHD, o_proj_HKD)
                   

    return x @ embedding_VD.T



## data loading ##
def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
        break
      result[i,j] = char
  return result

def input_to_output(np_array):
   zero_array = np.zeros( (BATCH_IN_SEQUENCES,SEQUENCE_LENGTH), dtype = jnp.uint8)
   zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0:SEQUENCE_LENGTH-1]
   return zero_array

def build_dataset():
    # Construct a tf.data.Dataset
    ds = tfds.load('lm1b', split='train', shuffle_files=False)

    # Build your input pipeline
    ds = ds.batch(BATCH_IN_SEQUENCES)
    return ds

def process_example(raw_example):
    numpy_strings = raw_example['text'].numpy()
    ascii_array_input = convert_to_ascii(numpy_strings, SEQUENCE_LENGTH)
    ascii_array_output = 0 * np.empty_like(ascii_array_input)
    ascii_array_output[:,0:SEQUENCE_LENGTH-1] = ascii_array_input[:, 1:SEQUENCE_LENGTH]
    return {"input" : jnp.asarray(ascii_array_input), "output" : jnp.asarray(ascii_array_output)}


def calculate_loss(params, model, inputs, outputs):
   proposed_outputs = model.apply(params, inputs)
   one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
   loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
   return jnp.mean(loss)

def step(state, model, inputs, outputs):
   loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs)
   state = state.apply_gradients(grads = grad)
   return loss, state

def main():
  mesh = jax.sharding.Mesh(np.reshape(  jax.devices(), (FSDP,TENSOR)), ["fsdp", "tp"])

  ds = tfds.load('lm1b', split='train', shuffle_files=False)
  ds = ds.batch(BATCH_IN_SEQUENCES)

  rngkey = jax.random.key(0)
  model = OurModel()

  shaped_init = jax.eval_shape( functools.partial(model.init, rngkey), jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))
  state_sharding = nn.get_sharding(shaped_init, mesh)
  _params = jax.jit(model.init, out_shardings = state_sharding)(rngkey, jax.ShapeDtypeStruct((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype = jnp.uint8))

  tx = optax.adam(learning_rate = LEARNING_RATE)
  state = train_state.TrainState.create(
    apply_fn = model.apply,
    params = _params,
    tx = tx
  )

  iter = 0
  static_step = jax.jit(step, static_argnums=1)

  last_step_time = time.time()
  stepnum = 0

  for example in ds:
    outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
    inputs = input_to_output(outputs)

    loss, state = static_step(state, model, inputs, outputs)
    stepnum += 1
    
    if stepnum % 10 == 0:
      new_time = time.time()
      time_elapsed_seconds = (new_time-last_step_time)
      last_step_time = new_time
      print(f"{iter} -> {loss} {time_elapsed_seconds}")
    
    iter += 1

if __name__ == "__main__":
   main()