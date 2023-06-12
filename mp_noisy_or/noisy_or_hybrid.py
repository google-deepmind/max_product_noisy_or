# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Our proposed hybrid approach where max-product is used to initialize VI."""

from mp_noisy_or import noisy_or_bp
from mp_noisy_or import noisy_or_vi


# pylint: disable=invalid-name
class NoisyOR_Hybrid:
  """Trains a NoisyOR model with the hybrid approach."""

  def __init__(self, config_BP, config_VI):
    # Seeds must match
    config_VI.seed = config_BP.seed
    if "seed" in config_BP.data.args:
      assert "seed" in config_VI.data.args
      config_VI.data.args.seed = config_BP.data.args.seed

    self.noisy_or_bp = noisy_or_bp.NoisyOR_BP(config=config_BP)
    self.noisy_or_vi = noisy_or_vi.NoisyOR_VI(config=config_VI)

  def train(self):
    results_BP = self.noisy_or_bp.train()
    log_potentials_BP = results_BP["log_potentials"]
    # The init Elbo will be evaluated
    results_VI = self.noisy_or_vi.train(log_potentials_BP)
    return results_BP, results_VI
