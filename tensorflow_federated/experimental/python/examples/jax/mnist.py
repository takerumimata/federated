# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example MNIST implementation in JAX."""

import collections
import jax
import numpy as np
import tensorflow_federated as tff

tff_params_type = tff.to_type(
    collections.OrderedDict([('weights', tff.TensorType(np.float32, (784, 10))),
                             ('bias', tff.TensorType(np.float32, (10,)))]))


def get_tff_batch_type(batch_size):
  return tff.to_type(
      collections.OrderedDict([
          ('pixels', tff.TensorType(np.float32, (batch_size, 784))),
          ('labels', tff.TensorType(np.float32, (batch_size,)))
      ]))


class Data(object):
  """Represents the MNIST data."""

  def __init__(self, batch_size, num_batches):
    self._batch_size = batch_size
    training_data, _ = tff.simulation.datasets.emnist.load_data()
    ds = training_data.create_tf_dataset_from_all_clients()
    self._ds = ds.batch(batch_size).take(num_batches)

  def generate(self):
    for element in iter(self._ds):
      pixels = element['pixels'].numpy().reshape(self._batch_size, -1)
      label = element['label'].numpy().astype(np.float32)
      yield collections.OrderedDict([('pixels', pixels), ('labels', label)])


def initialize():
  weights = jax.numpy.zeros((784, 10), dtype=np.float32)
  bias = jax.numpy.zeros((10,), dtype=np.float32)
  return collections.OrderedDict([('weights', weights), ('bias', bias)])


def loss(params, batch):
  y = jax.nn.softmax(
      jax.numpy.add(
          jax.numpy.matmul(batch['pixels'], params['weights']), params['bias']))
  targets = jax.nn.one_hot(jax.numpy.reshape(batch['labels'], -1), 10)
  return -jax.numpy.mean(jax.numpy.sum(targets * jax.numpy.log(y), axis=1))


def update(params, batch, step_size):
  grads = jax.api.grad(loss)(params, batch)
  return collections.OrderedDict([
      (k, params[k] - step_size * grads[k]) for k in ['weights', 'bias']
  ])
