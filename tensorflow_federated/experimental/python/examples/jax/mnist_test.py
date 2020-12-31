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

from absl.testing import absltest
import numpy as np
import tensorflow_federated as tff

from tensorflow_federated.experimental.python.examples.jax import mnist


class MnistTest(absltest.TestCase):

  # TODO(b/175888145): Evolve this into a complete federated training example.

  def setUp(self):
    super(MnistTest, self).setUp()
    self._batch_size = 50
    self._num_batches = 20
    self._data = mnist.Data(self._batch_size, self._num_batches)
    self._batch_type = mnist.get_tff_batch_type(self._batch_size)
    self._initialize_comp = tff.experimental.jax_computation(mnist.initialize)
    self._params_type = mnist.tff_params_type
    self._loss_comp = tff.experimental.jax_computation(
        mnist.loss, (self._params_type, self._batch_type))
    self._update_comp = tff.experimental.jax_computation(
        mnist.update, (self._params_type, self._batch_type, np.float32))

  def test_types(self):
    example_batch = next(self._data.generate())
    make_batch = tff.experimental.jax_computation(lambda: example_batch)
    self.assertEqual(
        str(make_batch.type_signature.result), str(self._batch_type))
    self.assertEqual(
        str(self._initialize_comp.type_signature.result),
        str(self._params_type))
    self.assertEqual(str(self._loss_comp.type_signature.result), 'float32')
    self.assertEqual(
        str(self._update_comp.type_signature.result), str(self._params_type))

  def test_training(self):

    def train_on_all_batches(params):
      for batch in self._data.generate():
        params = self._update_comp(params, batch, 0.001)
      return params

    def compute_average_loss(params):
      losses = []
      for batch in self._data.generate():
        losses.append(self._loss_comp(params, batch))
      return np.mean(losses)

    params = self._initialize_comp()
    initial_loss = compute_average_loss(params)
    for _ in range(3):
      params = train_on_all_batches(params)
      updated_loss = compute_average_loss(params)

    self.assertLess(updated_loss, initial_loss)


if __name__ == '__main__':
  tff.experimental.backends.xla.set_local_execution_context()
  absltest.main()
