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

"""Example script."""

import numpy as np

from mp_noisy_or import config
from mp_noisy_or import noisy_or_bp
from mp_noisy_or import results_utils

from absl import app

LP_THRE = np.log(2)


# pylint: disable=invalid-name
def run_bp_on_2D_blind_deconvolution(_):
  """Train a noisy OR Bayesian Network with Belief Propagation on the 2D blind deconvolution example and evaluate it."""
  # Train noisy-OR Bayesian network with BP
  this_config = config.get_config_BP_2Ddeconv()
  # Here, we modify the default parameters to accelerate convergence
  this_config.learning.num_iters = 600
  this_config.learning.proba_init = 0.9
  # Training should take 3min on a GPU
  NoisyOR = noisy_or_bp.NoisyOR_BP(this_config)
  results_BP = NoisyOR.train()

  # Extract the log-potentials
  log_potentials = np.array(results_BP["log_potentials"])[:5, :6, :6]
  W_learned = (log_potentials > LP_THRE).astype(float)

  # Compute metrics
  print(f"After {this_config.learning.num_iters} training iterations")
  # Test Elbo
  test_avg_elbo_mode = results_BP["all_test_avg_elbos_mode"][-1]
  print(f"Test elbo : {round(test_avg_elbo_mode, 3)}")
  # Test reconstruction error
  _, _, test_rec_ratio = results_utils.BD_reconstruction(
      NoisyOR.Xv_gt_test, results_BP["test_X_samples"], W_learned
  )
  print(f"Test rec. error: {round(100 *test_rec_ratio, 3)}%")


if __name__ == "__main__":
  app.run(run_bp_on_2D_blind_deconvolution)
