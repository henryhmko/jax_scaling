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
BATCH_IN_SEQUENCES = 384
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
  def __call__(self, x):
    '''
    input_tokens is [BATCH, SEQUENCE]
    '''
    embedding = self.param(
      'embedding',                # param name
      nn.initializers.normal(1),  # initializer func
      (VOCAB_DIM, EMBED_DIM),     # shape
      jnp.float32,                # dtype
    )

    x = embedding[x] #BATCH, SEQUENCE, EMBED
    # x = jnp.asarray(embedding)[input_tokens] #BATCH, SEQUENCE, EMBED

    for i in range(LAYERS):
      feedforward = self.param(
        'feedforward' + str(i),
        nn.initializers.lecun_normal(),
        (EMBED_DIM, FF_DIM),
        jnp.float32,
      )

      x = x @ feedforward # [B,S,E] @ [E, F] = [BATCH, SEQUENCE, FF_DIM]
      x = jax.nn.relu(x)  # why jax.nn.relu instead of flax.linen.relu?
      embed = self.param(
        'embed_' + str(i),
        nn.initializers.lecun_normal(),
        (FF_DIM, EMBED_DIM),
        jnp.float32,
      )
      x = x @ embed
      x = jax.nn.relu(x)

    return x @ embedding.T



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

def main():
   rngkey = jax.random.key(0)
   ds = build_dataset()
   model = OurModel()
   tx = optax.adam(learning_rate=LEARNING_RATE)
   _params = model.init(rngkey, jnp.ones((BATCH_IN_SEQUENCES, SEQUENCE_LENGTH), dtype=jnp.uint8))
   
   state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=_params,
      tx=tx # optimizer
   )

   step = 0
   for example in ds:
      outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
      inputs = input_to_output(outputs)
      
      loss, grad = jax.value_and_grad(calculate_loss)(state.params, model, inputs, outputs) # create a func that evaluates both calc_loss and gradient of it where calc_loss is the func to be differentiated
      state = state.apply_gradients(grads=grad) # updates step, params, opt_state, **kwargs in return value
      print(f"{step=}, {loss}")
      step += 1

if __name__ == "__main__":
   main()