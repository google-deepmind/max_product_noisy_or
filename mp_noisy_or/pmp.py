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


"""Run the PMP experiment on 2Ddeconv and BMF."""

import collections
import datetime

from ml_collections import config_flags
import numpy as np
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
import scipy
import tqdm

from mp_noisy_or import utils
from absl import app


_CONFIGS = config_flags.DEFINE_config_file(
    name="config",
    default="config.py",
    help_string="Training configuration",
)


# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield

###################################
######## 2D deconvolution #########
###################################

# From https://github.com/deepmind/PGMax/blob/main/examples/pmp_binary_deconvolution.ipynb
pW = 0.25
pS = 1e-75
pX = 1e-100


def PMP_2D_deconv(X_gt, is_training=True, W_learned=None):
  """Run the pertub-and-max-product algorithm for the 2D deconvolution experiment."""
  n_feat, feat_height, feat_width = 5, 6, 6

  if not is_training:
    assert W_learned is not None
    assert W_learned.shape == (n_feat, feat_height, feat_width)

  n_images, im_height, im_width = X_gt.shape
  s_height = im_height - feat_height + 1
  s_width = im_width - feat_width + 1

  ###################################
  ### Step 1: Create factor graph ###
  ###################################

  # Binary features
  W = vgroup.NDVarArray(num_states=2, shape=(n_feat, feat_height, feat_width))
  # Binary indicators of features locations
  S = vgroup.NDVarArray(
      num_states=2, shape=(n_images, n_feat, s_height, s_width)
  )
  # Auxiliary binary variables combining W and S
  SW = vgroup.NDVarArray(
      num_states=2,
      shape=(n_images, im_height, im_width, n_feat, feat_height, feat_width),
  )
  # Binary images obtained by convolution
  X = vgroup.NDVarArray(num_states=2, shape=X_gt.shape)

  # Factor graph
  fg = fgraph.FactorGraph(variable_groups=[S, W, SW, X])

  # Define the logical factors
  indices_for_ANDFactors = []
  variables_for_ANDFactors = []
  variables_for_ORFactors_dict = collections.defaultdict(list)
  for idx_img in tqdm.trange(n_images):
    for idx_s_height in range(s_height):
      for idx_s_width in range(s_width):
        for idx_feat in range(n_feat):
          for idx_feat_height in range(feat_height):
            for idx_feat_width in range(feat_width):
              idx_img_height = idx_feat_height + idx_s_height
              idx_img_width = idx_feat_width + idx_s_width

              # Store the relevant indices for the learni g part
              idx_S = (idx_img, idx_feat, idx_s_height, idx_s_width)
              idx_W = (idx_feat, idx_feat_height, idx_feat_width)
              idx_SW = (
                  idx_img,
                  idx_img_height,
                  idx_img_width,
                  idx_feat,
                  idx_feat_height,
                  idx_feat_width,
              )
              indices_for_ANDFactors.append([idx_S, idx_W, idx_SW])

              variables_for_ANDFactor = [
                  S[idx_S],
                  W[idx_W],
                  SW[idx_SW],
              ]
              variables_for_ANDFactors.append(variables_for_ANDFactor)

              X_var = X[idx_img, idx_img_height, idx_img_width]
              variables_for_ORFactors_dict[X_var].append(SW[idx_SW])

  # Define the ANDFactors
  AND_factor_group = fgroup.ANDFactorGroup(variables_for_ANDFactors)

  # Define the ORFactors
  variables_for_ORFactors = [
      list(tuple(variables_for_ORFactors_dict[X_var]) + (X_var,))
      for X_var in variables_for_ORFactors_dict
  ]

  # Add the two FactorGroups, which is computationally efficient
  OR_factor_group = fgroup.ORFactorGroup(variables_for_ORFactors)
  fg.add_factors([AND_factor_group, OR_factor_group])

  ###################################
  #### Step 2: Train noisy OR BP ####
  ###################################

  # BP functions
  bp = infer.BP(fg.bp_state, temperature=0.0)

  # Define the evidence
  uW = np.zeros((W.shape) + (2,))
  if is_training:
    uW[..., 1] = scipy.special.logit(pW)
    uW += np.random.gumbel(size=uW.shape)
  else:
    uW[..., 0] = (2 * W_learned - 1) * utils.CLIP_INF

  # Sparsity inducing priors for W and S
  uS = np.zeros((S.shape) + (2,))
  uS[..., 1] = scipy.special.logit(pS)

  # Likelihood the binary images given X
  uX = np.zeros((X_gt.shape) + (2,))
  uX[..., 0] = (2 * X_gt - 1) * scipy.special.logit(pX)

  # Run BP
  bp_arrays = bp.init(
      evidence_updates={
          S: uS + np.random.gumbel(size=uS.shape),
          W: uW,
          SW: np.zeros(shape=SW.shape),
          X: uX,
      }
  )
  bp_arrays = bp.run_bp(bp_arrays, num_iters=1_000, damping=0.5)
  beliefs = bp.get_beliefs(bp_arrays)
  map_states = infer.decode_map_states(beliefs)
  return map_states[S], map_states[W]


def run_2D_deconv(seed, ratio_train=0.8):
  """Run 2D deconvolution for one seed."""
  url = 'https://raw.githubusercontent.com/deepmind/PGMax/main/examples/example_data/conv_problem.npz'
  path = keras.utils.get_file('conv_problem.npz', url)
  data = np.load(path)
  Xv_gt = data["X"][:, 0, :, :]

  # Create train and test sets
  n_samples_train = int(ratio_train * len(Xv_gt))
  np.random.seed(seed)
  np.random.shuffle(Xv_gt)
  Xv_gt_train = Xv_gt[:n_samples_train]
  Xv_gt_test = Xv_gt[n_samples_train:]

  # Train
  S_train, W_learned = PMP_2D_deconv(Xv_gt_train, is_training=True)
  S_test, _ = PMP_2D_deconv(Xv_gt_test, is_training=False, W_learned=W_learned)
  return W_learned, Xv_gt_train, S_train, Xv_gt_test, S_test


###################################
############### BMF ###############
###################################


def PMP_BMF(X_gt, rank, p_UV, is_training=False, V_learned=None):
  """Run the pertub-and-max-product algorithm for the BMF experiment."""
  n_rows, n_cols = X_gt.shape

  if not is_training:
    assert V_learned is not None
    assert V_learned.shape == (rank, n_cols)

  ###################################
  ### Step 1: Create factor graph ###
  ###################################
  # Binary variables
  U = vgroup.NDVarArray(num_states=2, shape=(n_rows, rank))
  V = vgroup.NDVarArray(num_states=2, shape=(rank, n_cols))

  # Auxiliary binary variables combining U and V
  UV = vgroup.NDVarArray(num_states=2, shape=(n_rows, rank, n_cols))

  # Binary images obtained by convolution
  X = vgroup.NDVarArray(num_states=2, shape=X_gt.shape)

  # Factor graph
  fg = fgraph.FactorGraph(variable_groups=[U, V, UV, X])

  # Define the LogicalFactors
  variables_for_ANDFactors = []
  variables_for_ORFactors = []

  for idx_row in range(n_rows):
    for idx_col in range(n_cols):
      variables_for_ORFactor = []

      for idx_rank in range(rank):
        UV_var = UV[idx_row, idx_rank, idx_col]
        variables_for_ANDFactor = [
            U[idx_row, idx_rank],
            V[idx_rank, idx_col],
            UV_var,
        ]
        variables_for_ANDFactors.append(variables_for_ANDFactor)
        variables_for_ORFactor.append(UV_var)

      variables_for_ORFactor = list(
          tuple(variables_for_ORFactor) + (X[idx_row, idx_col],)
      )
      variables_for_ORFactors.append(variables_for_ORFactor)

  AND_factor_group = fgroup.ANDFactorGroup(variables_for_ANDFactors)
  OR_factor_group = fgroup.ORFactorGroup(variables_for_ORFactors)

  # Add the two FactorGroups, which is computationally efficient
  OR_factor_group = fgroup.ORFactorGroup(variables_for_ORFactors)
  fg.add_factors([AND_factor_group, OR_factor_group])

  ###################################
  #### Step 2: Train noisy OR BP ####
  ###################################

  # BP functions
  bp = infer.BP(fg.bp_state, temperature=0.0)

  # Define the evidence
  uV = np.zeros((V.shape) + (2,))
  if is_training:
    uV[..., 1] = scipy.special.logit(p_UV)
    uV += np.random.gumbel(size=uV.shape)
  else:
    uV[..., 0] = (2 * V_learned - 1) * utils.CLIP_INF

  # Sparsity inducing priors for W and S
  uU = np.zeros((U.shape) + (2,))
  uU[..., 1] = scipy.special.logit(p_UV)

  # Likelihood the binary images given X
  uX = np.zeros((X_gt.shape) + (2,))

  uX[..., 0] = (2 * X_gt - 1) * utils.CLIP_INF

  # Run BP
  bp_arrays = bp.init(
      evidence_updates={
          U: uU + np.random.gumbel(size=uU.shape),
          V: uV,
          UV: np.zeros(shape=UV.shape),
          X: uX,
      }
  )
  bp_arrays = bp.run_bp(bp_arrays, num_iters=1000, damping=0.5)
  beliefs = bp.get_beliefs(bp_arrays)
  map_states = infer.decode_map_states(beliefs)
  return map_states[U], map_states[V]


def run_BMF(seed, n_rows, rank, n_cols, p_Xon):
  """Run BMF for one seed."""
  np.random.seed(seed)

  # Note that p(Xv_ij=1) = p_X = 1 - (1 - p_UV** 2) ** rank
  p_UV = (1 - (1 - p_Xon) ** (1.0 / rank)) ** 0.5
  U_gt_test = np.random.binomial(n=1, p=p_UV, size=(n_rows, rank))
  U_gt_train = np.random.binomial(n=1, p=p_UV, size=(n_rows, rank))
  V_gt = np.random.binomial(n=1, p=p_UV, size=(rank, n_cols))

  Xv_gt_train = U_gt_train.dot(V_gt)
  Xv_gt_train[Xv_gt_train >= 1] = 1
  Xv_gt_test = U_gt_test.dot(V_gt)
  Xv_gt_test[Xv_gt_test >= 1] = 1

  # Train
  U_train, V_learned = PMP_BMF(Xv_gt_train, rank, p_UV, is_training=True)
  U_test, _ = PMP_BMF(
      Xv_gt_test, rank, p_UV, is_training=False, V_learned=V_learned
  )
  return V_learned, Xv_gt_train, U_train, Xv_gt_test, U_test


###################################
############## Train ###############
###################################


# pylint: disable=invalid-name
def train(_):
  """Train the noisy OR network on."""
  config = _CONFIGS.value
  dataset = config.dataset

  # First extract the config for the dataset
  if dataset == "BMF":
    config_PMP = config.config_PMP_BMF
    W_learned, Xv_gt_train, S_train, Xv_gt_test, S_test = run_BMF(**config_PMP)
  elif dataset == "2D_deconvolution":
    config_PMP = config.config_PMP_2Ddeconv
    W_learned, Xv_gt_train, S_train, Xv_gt_test, S_test = run_2D_deconv(
        **config_PMP
    )
  else:
    raise ValueError("Unknown dataset", dataset)


if __name__ == "__main__":
  app.run(train, load_cuda_libraries=False)
