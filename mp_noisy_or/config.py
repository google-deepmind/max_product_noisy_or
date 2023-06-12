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

"""Configuration to train the noisy OR models."""

import ml_collections


# pylint: disable=invalid-name
def get_config():
  """Training config for both BP and VI."""
  config = ml_collections.config_dict.ConfigDict()
  config.method = ""
  config.dataset = ""

  config.config_BP_20news = get_config_BP_20news()
  config.config_BP_BMF = get_config_BP_BMF()
  config.config_BP_2Ddeconv = get_config_BP_2Ddeconv()
  config.config_BP_yelp = get_config_BP_yelp()
  config.config_BP_imdb = get_config_BP_imdb()
  config.config_BP_abstract = get_config_BP_abstract()
  config.config_BP_agnews = get_config_BP_agnews()
  config.config_BP_patent = get_config_BP_patent()
  config.config_BP_overparam = get_config_BP_overparam()

  config.config_VI_20news = get_config_VI_20news()
  config.config_VI_BMF = get_config_VI_BMF()
  config.config_VI_2Ddeconv = get_config_VI_2Ddeconv()
  config.config_VI_yelp = get_config_VI_yelp()
  config.config_VI_imdb = get_config_VI_imdb()
  config.config_VI_abstract = get_config_VI_abstract()
  config.config_VI_agnews = get_config_VI_agnews()
  config.config_VI_patent = get_config_VI_patent()

  config.config_PMP_BMF = get_config_PMP_BMF()
  config.config_PMP_2Ddeconv = get_config_PMP_2Ddeconv()
  return config


##############################################
############## BP configs ####################
##############################################


def get_config_BP() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP."""
  config = ml_collections.config_dict.ConfigDict()
  config.seed = 0
  config.min_clip = 1e-5
  config.backend = "bp"

  config.data = ml_collections.config_dict.ConfigDict()
  config.data.args = ml_collections.config_dict.ConfigDict()

  config.bp = ml_collections.config_dict.ConfigDict()
  config.bp.temperature = 0.0  # max-product
  config.bp.num_iters = 100
  config.bp.damping = 0.5

  config.learning = ml_collections.config_dict.ConfigDict()
  # Initialization parameters
  config.learning.proba_init = 0.5
  config.learning.leak_proba_init = 0.9
  config.learning.leak_proba_init_not_updated = 0.99

  config.learning.noise_temperature_init = 0.0
  config.learning.learning_rate = 1e-2
  config.learning.num_iters = 1000
  config.learning.train_batch_size = 100_000
  config.learning.eval_every = 100
  config.learning.n_hidden_by_sample = 1
  config.learning.noise_temperature = 0.0

  config.inference = ml_collections.config_dict.ConfigDict()
  config.inference.store_inference_results = False
  config.inference.test_size_eval_and_store = 100_000
  config.inference.test_batch_size = 100_000

  # Mode at test time
  config.inference.n_hidden_by_sample = 1
  config.inference.noise_temperature = 0.0
  return config


def get_config_BP_20news() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on 20news."""
  config = get_config_BP()

  config.data.dataset = "20news"
  config.data.ratio_train = 0.7
  config.data.args.sparse_data = False
  config.data.args.n_layers = 3
  return config


def get_config_BP_yelp() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on yelp."""
  config = get_config_BP()

  # Train set is 560_000, test is 38_000
  config.data.dataset = "yelp_polarity_reviews"
  config.data.args.key_name = "text"
  config.data.args.vocab_size = 10_000
  config.data.args.max_sequence_length = 500
  config.data.args.n_layers = 5

  config.learning.learning_rate = 3e-4
  config.learning.train_batch_size = 128
  config.learning.num_iters = 3_600  # then VI training
  config.learning.eval_every = 600  # eval is slow

  config.inference.test_batch_size = 512
  return config


def get_config_BP_imdb() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on imdb."""
  config = get_config_BP_yelp()

  # Train set is 25_000, test is 25_000
  config.data.dataset = "imdb_reviews"
  return config


def get_config_BP_abstract() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on abstract."""
  config = get_config_BP_yelp()

  # Train set is 203_037, test is 6_440
  config.data.dataset = "scientific_papers"
  config.data.args.key_name = "abstract"
  return config


def get_config_BP_agnews() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on abstract."""
  config = get_config_BP_yelp()

  # Train set is 120_000, test is 7_600
  config.data.dataset = "ag_news_subset"
  config.data.args.key_name = "description"
  return config


def get_config_BP_patent() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on abstract."""
  config = get_config_BP_yelp()

  # Train set is 85_568, test is 4_754
  config.data.dataset = "patent"
  config.data.args.key_name = "description"
  return config


def get_config_BP_BMF() -> ml_collections.config_dict.ConfigDict:
  """Training config for BP on binary deconvolution."""
  config = get_config_BP()

  config.data.dataset = "BMF"
  config.data.ratio_train = 0.5
  config.data.args.seed = 0
  config.data.args.n_rows = 50
  config.data.args.rank = 15
  config.data.args.n_cols = 50
  config.data.args.p_Xon = 0.25

  config.learning.learning_rate = 1e-3
  config.learning.train_batch_size = 20
  config.learning.num_iters = 40_000
  config.learning.eval_every = 4_00
  config.inference.store_inference_results = True

  # Add noise to break symmetries
  config.learning.noise_temperature = 1.0
  config.learning.noise_temperature_init = 0.1
  return config


def get_config_BP_2Ddeconv() -> ml_collections.config_dict.ConfigDict:
  """Training config for BP on binary deconvolution."""
  config = get_config_BP()

  config.data.dataset = "2D_deconvolution"
  config.data.ratio_train = 0.8
  config.data.args.dataset_name = "pmp"
  config.data.args.W_shape = (16, 5, 5)

  config.learning.num_iters = 3000
  config.learning.eval_every = 300
  config.inference.store_inference_results = True

  # Add noise to break symmetries
  config.learning.noise_temperature = 1.0
  config.learning.noise_temperature_init = 0.1
  return config


def get_config_BP_overparam() -> ml_collections.config_dict.ConfigDict:
  """Training config for BP on binary deconvolution."""
  config = get_config_BP()

  config.data.dataset = "overparam"
  config.data.ratio_train = 0.9
  config.data.args.dataset_name = "PLNT"
  config.data.args.n_latent = 8

  config.learning.learning_rate = 1e-3  # as in the paper
  config.learning.train_batch_size = 20  # as in the paper
  config.learning.num_iters = 45_000  # as in the paper
  config.learning.eval_every = 5_000

  # Add noise to break symmetries
  config.learning.noise_temperature = 1.0
  config.learning.noise_temperature_init = 0.1
  return config


##############################################
############## VI configs ####################
##############################################


def get_config_VI() -> ml_collections.config_dict.ConfigDict:
  """Training config for the VI method."""
  config = ml_collections.config_dict.ConfigDict()
  config.seed = 0
  config.init_model_path = ""
  config.min_clip = 1e-5

  config.data = ml_collections.config_dict.ConfigDict()
  config.data.args = ml_collections.config_dict.ConfigDict()

  config.learning = ml_collections.config_dict.ConfigDict()
  # Initialization parameters
  config.learning.proba_init = 0.5
  config.learning.leak_proba_init = 0.9
  config.learning.leak_proba_init_not_updated = 0.99

  config.learning.noise_temperature_init = 0.0
  config.learning.n_inner_loops = 10
  config.learning.n_outer_loops = 10
  config.learning.learning_rate = 0.01
  config.learning.eval_every = 10
  config.learning.train_batch_size = 100_000
  config.learning.num_iters = 1000

  config.inference = ml_collections.config_dict.ConfigDict()
  config.inference.test_batch_size = 100_000
  config.inference.test_size_eval_and_store = 100_000
  config.inference.store_inference_results = False
  return config


def get_config_VI_20news() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on 20news."""
  config = get_config_VI()

  config.data.dataset = "20news"
  config.data.ratio_train = 0.7
  config.data.args.sparse_data = False
  config.data.args.n_layers = 3
  return config


def get_config_VI_20news_from_authors() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on 20news."""
  config = get_config_VI()

  config.data.dataset = "20news_from_authors"
  config.data.ratio_train = 0.7
  config.data.args.sparse_data = False
  return config


def get_config_VI_yelp() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on yelp."""
  config = get_config_VI()

  # Train set is 560_000, test is 38_000
  config.data.dataset = "yelp_polarity_reviews"
  config.data.args.key_name = "text"
  config.data.args.vocab_size = 10_000
  config.data.args.max_sequence_length = 500
  config.data.args.n_layers = 5

  config.learning.train_batch_size = 128
  config.learning.num_iters = 4_000
  config.learning.eval_every = 400

  config.inference.test_batch_size = 512
  return config


def get_config_VI_imdb() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on imdb."""
  config = get_config_VI_yelp()

  # Train set is 25_000, test is 25_000
  config.data.dataset = "imdb_reviews"
  return config


def get_config_VI_abstract() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on abstract."""
  config = get_config_VI_yelp()

  # Train set is 203_037, test is 6_440
  config.data.dataset = "scientific_papers"
  config.data.args.key_name = "abstract"
  return config


def get_config_VI_agnews() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on abstract."""
  config = get_config_VI_yelp()

  # Train set is 120_000, test is 7_600
  config.data.dataset = "ag_news_subset"
  config.data.args.key_name = "description"
  return config


def get_config_VI_patent() -> ml_collections.config_dict.ConfigDict:
  """Base config for BP on abstract."""
  config = get_config_VI_yelp()

  # Train set is 85_568, test is 4_754
  config.data.dataset = "patent"
  config.data.args.key_name = "description"
  return config


def get_config_VI_BMF() -> ml_collections.config_dict.ConfigDict:
  """Training config for BP on binary deconvolution."""
  config = get_config_VI()

  config.data.dataset = "BMF"
  config.data.ratio_train = 0.5
  config.data.args.seed = 0
  config.data.args.n_rows = 50
  config.data.args.rank = 15
  config.data.args.n_cols = 50
  config.data.args.p_Xon = 0.25

  config.learning.learning_rate = 1e-3
  config.learning.train_batch_size = 20
  config.learning.num_iters = 40_000
  config.learning.eval_every = 4_00
  config.inference.store_inference_results = True

  # Add noise to break symmetries
  config.learning.noise_temperature_init = 0.1
  return config


def get_config_VI_2Ddeconv() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on binary deconvolution."""
  config = get_config_VI()

  config.data.dataset = "2D_deconvolution"
  config.data.ratio_train = 0.8

  config.learning.num_iters = 3000
  config.learning.eval_every = 300
  config.inference.store_inference_results = True

  # Add noise to break symmetries
  config.learning.noise_temperature_init = 0.1
  return config


##############################################
############## PMP configs ###################
##############################################


def get_config_PMP_BMF() -> ml_collections.config_dict.ConfigDict:
  """Training config for BP on binary deconvolution."""
  config = ml_collections.config_dict.ConfigDict()
  config.seed = 0
  config.n_rows = 50
  config.rank = 15
  config.n_cols = 50
  config.p_Xon = 0.25
  return config


def get_config_PMP_2Ddeconv() -> ml_collections.config_dict.ConfigDict:
  """Training config for VI on binary deconvolution."""
  config = ml_collections.config_dict.ConfigDict()
  config.seed = 0
  config.ratio_train = 0.8
  return config
