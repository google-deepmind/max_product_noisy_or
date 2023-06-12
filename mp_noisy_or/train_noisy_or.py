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

"""Train the noisy OR models."""

from ml_collections import config_flags

from mp_noisy_or import noisy_or_bp
from mp_noisy_or import noisy_or_hybrid
from mp_noisy_or import noisy_or_vi
from absl import app

_CONFIGS = config_flags.DEFINE_config_file(
    name="config",
    default="config.py",
    help_string="Training configuration",
)


# pylint: disable=invalid-name
def train(_):
  """Train the noisy OR network on a dataset."""
  config = _CONFIGS.value
  dataset = config.dataset
  method = config.method

  # First extract the config for the dataset
  if dataset == "20news":
    config_BP = config.config_BP_20news
    config_VI = config.config_VI_20news
  elif dataset == "synthetic":
    config_BP = config.config_BP_synthetic
    config_VI = config.config_VI_synthetic
  elif dataset == "BMF":
    config_BP = config.config_BP_BMF
    config_VI = config.config_VI_BMF
  elif dataset == "2D_deconvolution":
    config_BP = config.config_BP_2Ddeconv
    config_VI = config.config_VI_2Ddeconv
  elif dataset == "overparam":
    config_BP = config.config_BP_overparam
    config_VI = None
  elif dataset == "yelp":
    config_BP = config.config_BP_yelp
    config_VI = config.config_VI_yelp
  elif dataset == "imdb":
    config_BP = config.config_BP_imdb
    config_VI = config.config_VI_imdb
  elif dataset == "abstract":
    config_BP = config.config_BP_abstract
    config_VI = config.config_VI_abstract
  elif dataset == "agnews":
    config_BP = config.config_BP_agnews
    config_VI = config.config_VI_agnews
  elif dataset == "patent":
    config_BP = config.config_BP_patent
    config_VI = config.config_VI_patent
  else:
    raise ValueError("Unknown dataset", dataset)

  # Second train the selected method
  if method == "BP":
    noisy_OR = noisy_or_bp.NoisyOR_BP(config_BP)
  elif method == "VI":
    noisy_OR = noisy_or_vi.NoisyOR_VI(config_VI)
  elif method == "hybrid":
    noisy_OR = noisy_or_hybrid.NoisyOR_Hybrid(
        config_BP=config_BP, config_VI=config_VI
    )
  else:
    raise ValueError("Unknown method", method)

  noisy_OR.train()


if __name__ == "__main__":
  app.run(train)
