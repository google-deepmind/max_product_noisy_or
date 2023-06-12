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

"""Defines data loader."""

import os

import numpy as np
import scipy.io as sio
from sklearn import cluster
from tensorflow import keras
import tensorflow_datasets as tfds

import pickle


DATA_FOLDER = "data/"
OVERPARAM_DATA_FOLDER = (
    "data/overparam/datasets/"
)


"""
To be used by our BP and VI training, a data loader must return:
  Xv_gt_train: the training data
  Xv_gt_test: the optional test data
  edges_children_to_parents: A dictionnary representing the noisy OR Bayesian network as
    {idx_child: {idx_parent: idx_potential}}
  X_shape: the shape of the unique array X containing all the hidden and visible
    variables
  log_potentials_shape: the shape of the unique array LP containing all the log
    potentials
  leak_potentials_mask: a mask indicating the potentials which connect a
    variable to the leak node
  dont_update_potentials_mask: a mask indicating the potentials that we do not
    want to update (often representing the noise probabilities)
  slice_visible: a slice indicating the visible variables in X
  slice_hidden: a slice indicating the hidden variables in X
  leak_node_idx: the index of the leak node in X
"""

# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-import-not-at-top
# pylint: disable=comparison-with-itself
# pylint: disable=g-complex-comprehension

#################################
###### Graph generation #########
#################################


def build_one_layer_or_factors(
    distmat, children_offset, n_clusters_by_node=5, ratio_children_parents=3
):
  """Build one layer of ORFactors in the hierarchy."""
  n_children = distmat.shape[0]
  n_parents = n_children // ratio_children_parents

  # First create n_parents clusters by hierarchical clustering
  clustering = cluster.AgglomerativeClustering(
      n_clusters=n_parents, affinity="precomputed", linkage="average"
  )
  clustering.fit(distmat)

  # Second add each node to a minimum of clusters
  # Compute the distance from nodes to clusters
  dist_to_clusters = np.zeros((n_children, n_parents))
  for idx_cluster in range(n_parents):
    this_cluster = np.where(clustering.labels_ == idx_cluster)[0]
    dist_to_cluster = distmat[this_cluster].mean(axis=0)
    dist_to_clusters[:, idx_cluster] = dist_to_cluster

  nodes_to_clusters = np.argsort(dist_to_clusters, axis=1)[
      :, :n_clusters_by_node
  ]
  # Compute the edges_children_to_parents
  edges_children_to_parents = {}
  for child_idx in range(n_children):
    cluster_idx = clustering.labels_[child_idx]
    closest_cluster_indices = list(nodes_to_clusters[child_idx])
    if cluster_idx not in closest_cluster_indices:
      # Rare case
      edges_children_to_parents[child_idx] = closest_cluster_indices + [
          cluster_idx
      ]
    else:
      edges_children_to_parents[child_idx] = closest_cluster_indices

  # Third, create the distance matrix for the next layer
  edges_p2c = {idx: [] for idx in range(n_parents)}
  for idx_child, idx_parents in edges_children_to_parents.items():
    for idx_parent in idx_parents:
      edges_p2c[idx_parent].append(idx_child)

  # Recompute the distance from nodes to each of the new clusters
  dist_to_clusters = np.zeros((n_children, n_parents))
  for idx_cluster in range(n_parents):
    this_cluster = np.array(edges_p2c[idx_cluster])
    dist_to_cluster = distmat[this_cluster].mean(axis=0)
    dist_to_clusters[:, idx_cluster] = dist_to_cluster

  distmat_next_layer = np.zeros((n_parents, n_parents))
  for idx_cluster in range(n_parents):
    this_cluster = np.array(edges_p2c[idx_cluster])
    dist_between_clusters = dist_to_clusters[this_cluster].mean(axis=0)
    distmat_next_layer[idx_cluster] = dist_between_clusters

  # Add offsets
  offset_edges_children_to_parents = {
      children_offset
      + idx_child: [
          children_offset + n_children + idx_parent
          for idx_parent in idx_parents
      ]
      for idx_child, idx_parents in edges_children_to_parents.items()
  }
  children_offset_next_layer = children_offset + n_children
  return (
      offset_edges_children_to_parents,
      distmat_next_layer,
      children_offset_next_layer,
  )


def build_or_factors(
    data_train,
    n_nodes_visible,
    n_layers,
    file_to_save=None,
    sparse_data=False,
    data_test=None,
):
  """Build a hierachical noisy OR Bayesian network."""
  print("Building the Bayesian network...")
  cooccurences = np.zeros((n_nodes_visible, n_nodes_visible))
  counts = np.zeros((n_nodes_visible,)) + 1e-8
  for data_row in data_train:
    # For tiny20 we have all the observations
    if len(data_row) == n_nodes_visible:
      nodes = np.where(data_row)[0]
    # For large datasets we only have the activations
    else:
      nodes = data_row
    for node in nodes:
      cooccurences[node, nodes] += 1
      counts[node] += 1

  # Normalize the counts
  # https://jmlr.org/papers/volume8/globerson07a/globerson07a.pdf
  norm_weights = cooccurences / counts.reshape(1, -1)
  norm_weights /= counts.reshape(-1, 1)
  norm_weights *= len(data_train)

  # Build the initial distance matrix
  distmat = np.exp(-norm_weights)
  np.fill_diagonal(distmat, 0)

  # Create the first layer
  (
      edges_children_to_parents,
      distmat_next_layer,
      offset_nodes_next_layer,
  ) = build_one_layer_or_factors(distmat, 0)

  # Create all the layers until the last one
  for _ in range(n_layers - 2):
    (
        new_edges_children_to_parents,
        distmat_next_layer,
        offset_nodes_next_layer,
    ) = build_one_layer_or_factors(distmat_next_layer, offset_nodes_next_layer)
    for k, v in new_edges_children_to_parents.items():
      edges_children_to_parents[k] = v

  # Create the last layer
  n_children = distmat_next_layer.shape[0]
  leak_node_idx = offset_nodes_next_layer + n_children
  for idx_child in range(n_children):
    edges_children_to_parents[offset_nodes_next_layer + idx_child] = [
        leak_node_idx
    ]

  # Add potential index and value
  idx_potential = 0
  edges_children_to_parents_augmented = {}
  leak_potentials_mask = []

  for idx_child, idx_parents in edges_children_to_parents.items():
    edges_children_to_parents_augmented[idx_child] = {}
    for idx_parent in idx_parents:
      edges_children_to_parents_augmented[idx_child][idx_parent] = idx_potential
      idx_potential += 1

      leak_potentials_mask.append(int(idx_parent == leak_node_idx))

    # Connect to leak node
    if leak_node_idx not in idx_parents:
      edges_children_to_parents_augmented[idx_child][
          leak_node_idx
      ] = idx_potential
      idx_potential += 1

      leak_potentials_mask.append(1)

  # Return the required quantities
  log_potentials_shape = (idx_potential,)
  n_nodes = max(edges_children_to_parents.keys()) + 2  # add leak node
  X_shape = (n_nodes,)
  slice_visible = np.s_[:n_nodes_visible]
  slice_hidden = np.s_[n_nodes_visible : n_nodes - 1]
  assert leak_node_idx == n_nodes - 1

  if sparse_data:
    data_train = [np.where(data_row != 0)[0] for data_row in data_train]
    if data_test is not None:
      data_test = [np.where(data_row != 0)[0] for data_row in data_test]

  # Optionally save
  if file_to_save is not None:
    data_to_save = {
        "Xv_gt_train": data_train,
        "Xv_gt_test": data_test,
        "edges_children_to_parents": edges_children_to_parents_augmented,
        "X_shape": X_shape,
        "log_potentials_shape": log_potentials_shape,
        "leak_potentials_mask": leak_potentials_mask,
        "slice_visible": slice_visible,
        "slice_hidden": slice_hidden,
        "leak_node_idx": leak_node_idx,
    }
    print("Saving processed data at {}...".format(file_to_save))
    with open(file_to_save, "wb") as f:
      pickle.dump(data_to_save, f)

  return (
      data_train,
      data_test,
      edges_children_to_parents_augmented,
      X_shape,
      log_potentials_shape,
      leak_potentials_mask,
      None,
      slice_visible,
      slice_hidden,
      leak_node_idx,
  )


def train_test_shuffle_split(Xv_gt, seed, ratio_train):
  """Split train and test."""
  np.random.seed(seed)
  assert ratio_train <= 1
  # Random train and test: shuffle first, then split
  n_samples_train = int(ratio_train * len(Xv_gt))
  np.random.shuffle(Xv_gt)
  Xv_gt_train = Xv_gt[:n_samples_train]
  Xv_gt_test = Xv_gt[n_samples_train:]
  return Xv_gt_train, Xv_gt_test


#################################
####### Tiny20 dataset ##########
#################################


def load_20news_w100(n_layers, sparse_data=False):
  """Load the data for the tiny20 dataset."""
  # Load dataset and words
  filename = DATA_FOLDER + "20news_w100.mat"
  with open(filename, "rb") as f:
    data = sio.loadmat(f)

  documents = data["documents"].todense()
  Xv_gt = np.array(documents.T.astype(float))
  n_words = Xv_gt.shape[1]
  return build_or_factors(Xv_gt, n_words, n_layers, sparse_data=sparse_data)


#################################
###### Tensorflow datasets ######
#################################


def load_yelp_dataset(**kwargs):
  return load_large_datasets(dataset="yelp_polarity_reviews", **kwargs)


def load_imdb_dataset(**kwargs):
  return load_large_datasets(dataset="imdb_reviews", **kwargs)


def load_abstract_dataset(**kwargs):
  return load_large_datasets(dataset="scientific_papers", **kwargs)


def load_agnews_dataset(**kwargs):
  return load_large_datasets(dataset="ag_news_subset", **kwargs)


def load_patent_dataset(**kwargs):
  return load_large_datasets(dataset="big_patent/f", **kwargs)


def load_large_datasets(
    dataset, key_name, vocab_size, max_sequence_length, n_layers
):
  """Load data for large Tensorflow datasets."""
  # https://www.tensorflow.org/datasets/catalog/scientific_papers
  filename = (
      DATA_FOLDER
      + "{}_vocabsize_{}_nlayers_{}_maxseqlength{}.npz".format(
          dataset, vocab_size, n_layers, max_sequence_length
      )
  )
  if os.path.exists(filename):
    print("Loading processed data at {}...".format(filename))
    data = pickle.load(filename)
    return (
        data["Xv_gt_train"],
        data["Xv_gt_test"],
        data["edges_children_to_parents"],
        data["X_shape"],
        data["log_potentials_shape"],
        data["leak_potentials_mask"],
        None,
        data["slice_visible"],
        data["slice_hidden"],
        data["leak_node_idx"],
    )

  # Training set
  data_train = tfds.load(dataset, split="train", batch_size=-1)
  data_train = tfds.as_numpy(data_train)[key_name]

  data_test = tfds.load(dataset, split="test", batch_size=-1)
  data_test = tfds.as_numpy(data_test)[key_name]

  # Define the vectorizer on the training data
  # https://www.tensorflow.org/tutorials/load_data/text
  vectorize_layer = keras.layers.TextVectorization(
      max_tokens=vocab_size,
      output_mode="int",
      output_sequence_length=max_sequence_length,
  )
  vectorize_layer.adapt(data_train)

  data_train = np.array(vectorize_layer(data_train))
  data_test = np.array(vectorize_layer(data_test))
  # vectorize_layer.get_vocabulary() gives the words
  print(vectorize_layer.get_vocabulary()[:100])
  print("Data train shape: ", data_train.shape)
  print("Data test shape: ", data_test.shape)

  train_binaries = []
  for train_row in data_train:
    unique_words = np.unique(train_row)
    # Remove elements 0 and 1, which are '' and UNK
    unique_words = unique_words[
        np.logical_and(unique_words != 0, unique_words != 1)
    ]
    train_binaries.append(unique_words)

  test_binaries = []
  for test_row in data_test:
    unique_words = np.unique(test_row)
    # Remove elements 0 and 1, which are '' and UNK
    unique_words = unique_words[
        np.logical_and(unique_words != 0, unique_words != 1)
    ]
    test_binaries.append(unique_words)

  # Build the OR factor
  return build_or_factors(
      data_train=train_binaries,
      n_nodes_visible=vocab_size,
      n_layers=n_layers,
      file_to_save=filename,
      data_test=test_binaries,
      sparse_data=False,  # data is already sparsified
  )


#################################
#### Binary deconvolution #######
#################################


def load_binary_deconvolution_data(dataset_name="pmp", W_shape=(16, 3, 3)):
  """Load data for binary deconvolution."""
  if dataset_name == "pmp":
    # Load the data from the PMP paper
    url = 'https://raw.githubusercontent.com/deepmind/PGMax/main/examples/example_data/conv_problem.npz'
    path = keras.utils.get_file('conv_problem.npz', url)
    data = np.load(path)
    W_gt = data["W"][0]
    X_gt_train = data["X"][:, 0, :, :]
    X_gt_test = None

    # Augment the parameters as in the PMP paper
    n_feat, feat_height, feat_width = W_gt.shape
    n_feat += 1
    feat_height += 1
    feat_width += 1

  else:
    raise ValueError("Unknown dataset", dataset_name)

  _, im_height, im_width = X_gt_train.shape
  assert im_height == im_width
  s_height = im_height - feat_height + 1
  s_width = im_width - feat_width + 1

  lp_shape = max([n_feat, feat_height, feat_width])
  # The log-potentials LP are such that
  # LP[:n_feat, :feat_height, :feat_width] give the failure probabilities
  # LP[n_feat, 0, :n_feat] give the prior of the latent variables
  # LP[n_feat, lp_shape - 1, lp_shape - 1] give the shared noise probability
  log_potentials_shape = (n_feat + 1, lp_shape + 1, lp_shape + 1)

  # The prior and failure probabilities are initialized differently
  leak_potentials_mask = np.zeros(shape=log_potentials_shape)
  leak_potentials_mask[n_feat, 0, :n_feat] = 1

  # The noise probability is fixed during training
  dont_update_potentials_mask = np.zeros(shape=log_potentials_shape)
  dont_update_potentials_mask[n_feat, lp_shape, lp_shape] = 1

  # The variables X are such that
  # X[:n_feat, :s_height, :s_width] corresponds to the hidden variables Xh
  # X[n_feat, :im_height, :im_width] corresponds to the visible variables Xv
  # X[n_feat, im_height, im_width] is the leak node
  X_shape = (n_feat + 1, im_height + 1, im_width + 1)
  slice_visible = np.s_[n_feat, :im_height, :im_width]
  slice_hidden = np.s_[:n_feat, :s_height, :s_width]
  leak_node_idx = (n_feat, im_height, im_width)

  edges_children_to_parents = {}
  for idx_s_height in range(s_height):
    for idx_s_width in range(s_width):
      for idx_feat in range(n_feat):
        # First, connect each hidden variable to the leak node
        # with a feature-specific prior probability
        edges_children_to_parents[(idx_feat, idx_s_height, idx_s_width)] = {
            leak_node_idx: (n_feat, 0, idx_feat)
        }

        # Second, consider edges where the child is a visible variable
        for idx_feat_height in range(feat_height):
          for idx_feat_width in range(feat_width):
            idx_img_height = idx_feat_height + idx_s_height
            idx_img_width = idx_feat_width + idx_s_width

            # Connect each visible variable to the leak node
            # with shared noise probability
            if (
                n_feat,
                idx_img_height,
                idx_img_width,
            ) not in edges_children_to_parents:
              edges_children_to_parents[
                  (n_feat, idx_img_height, idx_img_width)
              ] = {leak_node_idx: (n_feat, lp_shape, lp_shape)}

            # Connect each visible variable to a hidden variable
            # Format {idx_child: {idx_parent: idx_potential}}
            edges_children_to_parents[(n_feat, idx_img_height, idx_img_width)][
                (idx_feat, idx_s_height, idx_s_width)
            ] = (idx_feat, idx_feat_height, idx_feat_width)

  return (
      X_gt_train,
      X_gt_test,
      edges_children_to_parents,
      X_shape,
      log_potentials_shape,
      leak_potentials_mask,
      dont_update_potentials_mask,
      slice_visible,
      slice_hidden,
      leak_node_idx,
  )

#################################
## Binary matrix factorization ##
#################################


def load_BMF_data(seed, n_rows, rank, n_cols, p_Xon):
  """Generate the Binary Matrix Factorization data."""
  np.random.seed(seed)

  # Note that p(Xv_ij=1) = p_X = 1 - (1 - p_UV** 2) ** rank
  p_UV = (1 - (1 - p_Xon) ** (1.0 / rank)) ** 0.5
  U_gt_test = np.random.binomial(n=1, p=p_UV, size=(n_rows, rank))
  U_gt_train = np.random.binomial(n=1, p=p_UV, size=(n_rows, rank))
  V_gt = np.random.binomial(n=1, p=p_UV, size=(rank, n_cols))

  Xv_gt_train = U_gt_train.dot(V_gt)
  Xv_gt_train[Xv_gt_train >= 1] = 1

  print("Average number of activations in X: ", Xv_gt_train.mean())
  Xv_gt_test = U_gt_test.dot(V_gt)
  Xv_gt_test[Xv_gt_test >= 1] = 1

  # The log-potentials LP are such that
  # LP[:rank, :n_cols] give the failure probabilities
  # LP[rank, 0] give the shared noise probability
  # LP[rank, n_cols] give the shared noise probability
  log_potentials_shape = (rank + 1, n_cols + 1)

  # The prior and failure probabilities are initialized differently
  leak_potentials_mask = np.zeros(shape=log_potentials_shape)
  leak_potentials_mask[rank, :n_cols] = 1

  # The noise probability is fixed during training
  dont_update_potentials_mask = np.zeros(shape=log_potentials_shape)
  dont_update_potentials_mask[rank, n_cols] = 1

  # The variables X are such that
  # X[0, :rank] corresponds to the hidden variables U
  # X[1, :n_cols] corresponds to the visible variables Xv
  # X[1, n_cols] is the leak node
  X_shape = (2, n_cols + 1)
  slice_hidden = np.s_[0, :rank]
  slice_visible = np.s_[1, :n_cols]
  leak_node_idx = (1, n_cols)

  edges_children_to_parents = {}
  for idx_rank in range(rank):
    # Connect each hidden to the leak node with a shared prior probability
    edges_children_to_parents[(0, idx_rank)] = {leak_node_idx: (rank, 0)}

    # Second consider edges where the child is a visible variable
    for idx_col in range(n_cols):
      if (1, idx_col) not in edges_children_to_parents:
        # Connect each hidden to the leak node with a shared noise probability
        edges_children_to_parents[(1, idx_col)] = {
            leak_node_idx: (rank, n_cols)
        }

      # Connect each visible variable to a hidden variable
      # Format {idx_child: {idx_parent: idx_potential}}
      edges_children_to_parents[(1, idx_col)][(0, idx_rank)] = (
          idx_rank,
          idx_col,
      )

  return (
      Xv_gt_train,
      Xv_gt_test,
      edges_children_to_parents,
      X_shape,
      log_potentials_shape,
      leak_potentials_mask,
      dont_update_potentials_mask,
      slice_visible,
      slice_hidden,
      leak_node_idx,
      p_UV,
  )


#################################
###### Overparam datasets #######
#################################


def load_overparam_data(dataset_name, n_latent=8, img_size=64):
  """Load the data for the overparam experiments."""
  assert n_latent < img_size

  if dataset_name == "IMG-FLIP":
    filename = (
        OVERPARAM_DATA_FOLDER + dataset_name + "/samples/samples_str10percent"
    )
  else:
    filename = (
        OVERPARAM_DATA_FOLDER + dataset_name + "/samples/raw_samples_n10000_s0"
    )
  X_gt = np.loadtxt(open(filename, "rb"))
  img_size = X_gt.shape[1]

  # The log-potentials LP are such that
  # LP[:n_latent, :img_size] give the failure probabilities
  # LP[n_latent, :n_latent] give each hidden prior probability
  # LP[[n_latent, img_size] give the shared noise probability
  log_potentials_shape = (n_latent + 1, img_size + 1)

  # The prior and failure probabilities are initialized differently
  leak_potentials_mask = np.zeros(shape=log_potentials_shape)
  leak_potentials_mask[n_latent, :-1] = 1

  # The noise probability is fixed during training
  dont_update_potentials_mask = np.zeros(shape=log_potentials_shape)
  dont_update_potentials_mask[n_latent, img_size] = 1

  # The variables X are such that
  # X[0, :n_latent] corresponds to the hidden variables Xh
  # X[1, :img_size] corresponds to the visible variables Xv
  # X[0, img_size] is the leak node
  X_shape = (2, img_size + 1)
  slice_hidden = np.s_[0, :n_latent]
  slice_visible = np.s_[1, :img_size]
  leak_node_idx = (0, img_size)

  edges_children_to_parents = {}
  for idx_latent in range(n_latent):
    # Connect each hidden to the leak node with the hidden prior probability
    edges_children_to_parents[(0, idx_latent)] = {
        leak_node_idx: (n_latent, idx_latent)
    }

    # Second consider edges where the child is a visible variable
    for idx_pixel in range(img_size):
      if (1, idx_pixel) not in edges_children_to_parents:
        # Connect each hidden to the leak node with a shared prior probability
        edges_children_to_parents[(1, idx_pixel)] = {
            leak_node_idx: (n_latent, img_size)
        }

      # Connect each visible variable to a hidden variable
      # Format {idx_child: {idx_parent: idx_potential}}
      edges_children_to_parents[(1, idx_pixel)][(0, idx_latent)] = (
          idx_latent,
          idx_pixel,
      )

  return (
      X_gt,
      None,
      edges_children_to_parents,
      X_shape,
      log_potentials_shape,
      leak_potentials_mask,
      dont_update_potentials_mask,
      slice_visible,
      slice_hidden,
      leak_node_idx,
  )


DATA_LOADER = {
    "20news": load_20news_w100,
    "BMF": load_BMF_data,
    "2D_deconvolution": load_binary_deconvolution_data,
    "yelp_polarity_reviews": load_yelp_dataset,
    "imdb_reviews": load_imdb_dataset,
    "scientific_papers": load_abstract_dataset,
    "ag_news_subset": load_agnews_dataset,
    "patent": load_patent_dataset,
    "overparam": load_overparam_data,
}
