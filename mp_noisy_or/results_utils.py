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

"""Utils functions for computing the results."""

import jax
from jax.lax import dynamic_slice
from jax.lax import pad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy




def plot_images(
    images,
    display=True,
    nr=None,
    images_min=None,
    images_max=None,
    colorbar=False,
):
  """Useful function for visualizing several images."""
  n_images, H, W = images.shape
  if images_min is not None:
    images_min = min(images_min, images.min())
  else:
    images_min = images.min()
  images = images - images_min

  if images_max is not None:
    images_max = max(images_max, images.max())
  else:
    images_max = images.max()
  images /= images_max + 1e-10

  if nr is None:
    nr = nc = np.ceil(np.sqrt(n_images)).astype(int)
  else:
    nc = (n_images + nr - 1) // nr

  big_image = np.ones(((H + 1) * nr + 1, (W + 1) * nc + 1, 3))
  big_image[..., :3] = 0
  big_image[:: H + 1] = [0.5, 0, 0.5]

  im = 0
  for r in range(nr):
    for c in range(nc):
      if im < n_images:
        big_image[
            (H + 1) * r + 1 : (H + 1) * r + 1 + H,
            (W + 1) * c + 1 : (W + 1) * c + 1 + W,
            :,
        ] = images[im, :, :, None]
        im += 1

  if display:
    plt.figure(figsize=(10, 10))
    plt.imshow(big_image, interpolation="none")
    plt.axis("off")
    if colorbar:
      plt.colorbar()
  return big_image


#################################
#### Binary deconvolution #######
#################################


def BMF_metrics(
    Xv_gt, X_samples, log_potentials, config, log_potentials_threshold
):
  """Process the BMF outputs and compute the reconstruction error."""
  # Threshold the log potentials
  data_config = config.data.args
  V_bp = log_potentials[: data_config.rank, : data_config.n_cols]
  V_bp = (V_bp >= log_potentials_threshold).astype(float)

  # Extract the inferred hidden
  U_bp = X_samples[:, 0, : data_config.rank]
  # print("Avg hidden activations", U_bp.mean())
  return BMF_reconstruction(Xv_gt, U_bp, V_bp)


def BMF_reconstruction(Xv_gt, U_bp, V_bp):
  """Reconstruction error for BMF."""
  # Reconstruct X
  X_bp = np.array(U_bp.dot(V_bp))
  X_bp[X_bp >= 1] = 1
  rec_ratio_X = np.abs(X_bp - Xv_gt).sum() / Xv_gt.size
  return rec_ratio_X


#################################
#### OVERPARAMETRIZATION ########
#################################

# Params from the paper
PRIOR_THRESHOLD = 0.02
MATCHING_THRESHOLD = 1


def count_gt_recovered(Xh_gt, log_potentials_BP):
  """Count the nujber of GT parameters recovered."""
  n_latent = log_potentials_BP.shape[0] - 1
  # print("prior", log_potentials_BP[-1, -1], log_potentials_BP[-1, :n_latent])
  priors_BP = 1 - np.exp(-log_potentials_BP)[-1, :n_latent]
  keep = priors_BP > PRIOR_THRESHOLD

  log_potentials_BP_filtered = log_potentials_BP[:n_latent, :-1][keep]
  n_latent_filtered = log_potentials_BP_filtered.shape[0]

  n_gt = Xh_gt.shape[0]
  log_potentials_gt = -np.log(Xh_gt)
  matching_cost = np.zeros((n_gt, n_latent_filtered))
  for idx_latent in range(n_latent_filtered):
    for idx_gt in range(n_gt):
      matching_cost[idx_gt, idx_latent] = np.max(
          np.abs(
              log_potentials_BP_filtered[idx_latent] - log_potentials_gt[idx_gt]
          )
      )

  rows_indices, cols_indices = scipy.optimize.linear_sum_assignment(
      matching_cost
  )
  n_matched = sum(
      matching_cost[rows_indices, cols_indices] <= MATCHING_THRESHOLD
  )
  return n_matched


#################################
#### BLIND DECONVOLUTION ########
#################################


@jax.jit
def or_layer_jax(S, W):
  """Jax convolution of S and W for 2D BD."""
  _, n_feat, s_height, s_width = S.shape
  _, n_feat, feat_height, feat_width = W.shape
  im_height, im_width = s_height + feat_height - 1, s_width + feat_width - 1

  # Revert the features to have the proper orientations
  Wrev = W[:, :, ::-1, ::-1]

  # Pad the feature locations
  Spad = pad(
      S,
      0.0,
      (
          (0, 0, 0),
          (0, 0, 0),
          (feat_height - 1, feat_height - 1, 0),
          (feat_width - 1, feat_width - 1, 0),
      ),
  )

  # Convolve Spad and W
  def compute_sample(Spad1):
    def compute_pixel(r, c):
      X1 = (
          1
          - dynamic_slice(Spad1, (0, r, c), (n_feat, feat_height, feat_width))
          * Wrev
      ).prod((1, 2, 3))
      return 1 - X1

    compute_cols = jax.vmap(compute_pixel, in_axes=(None, 0), out_axes=1)
    compute_rows_cols = jax.vmap(compute_cols, in_axes=(0, None), out_axes=1)
    return compute_rows_cols(jnp.arange(im_height), jnp.arange(im_width))

  return jax.vmap(compute_sample, in_axes=0, out_axes=0)(Spad)


def BD_reconstruction(Xv_gt_test, test_X_samples, log_potentials_thre):
  """Reconstruction error for BD."""
  if log_potentials_thre.ndim == 3:
    _, im_height, im_width = Xv_gt_test.shape
    n_feat, feat_height, feat_width = log_potentials_thre.shape
    s_height = im_height - feat_height + 1
    s_width = im_width - feat_width + 1
    feats_activations = test_X_samples[:, :n_feat, :s_height, :s_width].astype(
        float
    )
    rec_X_test = or_layer_jax(feats_activations, log_potentials_thre[None])[
        :, 0
    ]

  elif log_potentials_thre.ndim == 4:
    _, _, im_height, im_width = Xv_gt_test.shape
    _, n_feat, feat_height, feat_width = log_potentials_thre.shape
    s_height = im_height - feat_height + 1
    s_width = im_width - feat_width + 1
    feats_activations = test_X_samples[:, :n_feat, :s_height, :s_width].astype(
        float
    )
    rec_X_test = or_layer_jax(feats_activations, log_potentials_thre)

  rec_ratio = np.abs(Xv_gt_test != rec_X_test).sum() / rec_X_test.size
  return feats_activations, rec_X_test, rec_ratio


def iou(a, b):
  return np.logical_and(a, b).sum() / np.logical_or(a, b).sum()


def features_iou(W_gt, log_potentials_thre):
  """Compute the features IOU."""
  assert log_potentials_thre.shape == (5, 6, 6)

  n_gt = W_gt.shape[0]
  n_log_potentials = log_potentials_thre.shape[0]

  matching_costs = np.zeros((n_gt, n_log_potentials))
  for idx_latent in range(n_log_potentials):
    for idx_gt in range(n_gt):
      # List all the options
      matching_cost_options = []
      for offset_r in range(2):
        for offset_c in range(2):
          matching_cost_options.append(
              iou(
                  log_potentials_thre[
                      idx_latent,
                      offset_r : offset_r + 5,
                      offset_c : offset_c + 5,
                  ],
                  W_gt[idx_gt],
              )
          )
      matching_costs[idx_gt, idx_latent] = np.max(matching_cost_options)

  # Hungarian matching
  rows_indices, cols_indices = scipy.optimize.linear_sum_assignment(
      -matching_costs
  )
  # print(matching_costs[rows_indices, cols_indices])
  return np.mean(matching_costs[rows_indices, cols_indices])


def visualize_cuts(feats):
  """Visualize cuts."""
  assert feats.ndim == 4
  n_samples, _, img_width, img_height = feats.shape
  vup, vdown, hup, hdown = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
  images = np.zeros((n_samples, 2 * img_width, 2 * img_height), float)

  # image[1::2, ::2] = 0.5 + np.where(vup > vdown, vup * 0.5, vdown * -0.5)
  # image[::2, 1::2] = 0.5 + np.where(hup > hdown, hup * 0.5, hdown * -0.5)
  images[:, 1::2, ::2] += vup + 2 * vdown
  images[:, ::2, 1::2] -= hup + 2 * hdown
  return images
