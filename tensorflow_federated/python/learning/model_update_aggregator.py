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
"""Generic aggregator for model updates in federated averaging."""

import math

import tensorflow_privacy as tfp

from tensorflow_federated.python.aggregators import clipping_factory
from tensorflow_federated.python.aggregators import dp_factory
from tensorflow_federated.python.aggregators import factory
from tensorflow_federated.python.aggregators import mean_factory
from tensorflow_federated.python.aggregators import quantile_estimation


def robust_aggregator(
    zeroing: bool = True,
    clipping: bool = True) -> factory.WeightedAggregationFactory:
  """Creates aggregator for mean with adaptive zeroing and clipping.

  Zeroes out extremely large values for robustness to data corruption on
  devices, and clips to moderately high norm for robustness to outliers.

  Args:
    zeroing: Whether to enable adaptive zeroing.
    clipping: Whether to enable adaptive clipping.

  Returns:
    A `factory.WeightedAggregationFactory` with zeroing and clipping.
  """
  factory_ = mean_factory.MeanFactory()

  if clipping:
    # Adapts relatively quickly to a moderately high norm.
    clipping_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=1.0, target_quantile=0.8, learning_rate=0.2)
    factory_ = clipping_factory.ClippingFactory(clipping_norm, factory_)

  if zeroing:
    # Adapts very quickly to a value somewhat higher than the highest
    # values so far seen.
    zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=10.0,
        target_quantile=0.98,
        learning_rate=math.log(10.0),
        multiplier=2.0,
        increment=1.0)
    factory_ = clipping_factory.ZeroingFactory(zeroing_norm, factory_)

  return factory_


def dp_aggregator(noise_multiplier: float,
                  clients_per_round: float,
                  zeroing: bool = True) -> factory.UnweightedAggregationFactory:
  """Creates aggregator with adaptive zeroing and differential privacy.

  Zeroes out extremely large values for robustness to data corruption.

  Args:
    noise_multiplier: A float specifying the noise multiplier for the Gaussian
      mechanism for model updates.
    clients_per_round: A float specifying the expected number of clients per
      round.
    zeroing: Whether to enable adaptive zeroing.

  Returns:
    A `factory.WeightedAggregationFactory` with zeroing and differential
      privacy.
  """

  # clipped_count_stddev defaults to 0.05 * clients_per_round. The noised
  # fraction of unclipped updates will be within 0.1 of the true fraction with
  # of unclipped updates will be within 0.1 of the true fraction with
  # 95.4% probability, and will be within 0.15 of the true fraction with
  # 99.7% probability. Even in this unlikely case, the error on the update
  # would be a factor of exp(0.2 * 0.15) = 1.03, a small deviation. So this
  # default gives maximal privacy for acceptable probability of deviation.
  clipped_count_stddev = 0.05 * clients_per_round

  query = tfp.QuantileAdaptiveClipAverageQuery(
      initial_l2_norm_clip=0.1,
      noise_multiplier=noise_multiplier,
      denominator=clients_per_round,
      target_unclipped_quantile=0.5,
      learning_rate=0.2,
      clipped_count_stddev=clipped_count_stddev,
      expected_num_records=clients_per_round,
      geometric_update=True)

  factory_ = dp_factory.DifferentiallyPrivateFactory(query)

  if zeroing:
    # Adapts very quickly to a value somewhat higher than the highest
    # values so far seen.
    zeroing_norm = quantile_estimation.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=10.0,
        target_quantile=0.98,
        learning_rate=math.log(10.0),
        multiplier=2.0,
        increment=1.0)
    factory_ = clipping_factory.ZeroingFactory(zeroing_norm, factory_)

  return factory_
