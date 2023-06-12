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

"""Utils functions."""

import jax
import jax.numpy as jnp
import numpy as np


CLIP_INF = -1e6


# pylint: disable=invalid-name
# pylint: disable=g-explicit-length-test
@jax.jit
def log1mexp(x):
  """Stable implementation of f(x) = log(1 - exp(-x)) for x >= 0 following https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf."""
  y = jnp.where(
      x <= jnp.log(2),
      jnp.log(-jnp.expm1(-x)),
      jnp.log1p(-jnp.exp(-x))
  )
  return jnp.clip(y, CLIP_INF, None)


@jax.jit
def get_unique_masks_locations_counts(array):
  """Jit compatible in-house implementations of jnp.unique."""
  n_rows = array.shape[0]
  masks = jnp.zeros((n_rows,) + array.shape)
  for idx in range(array.shape[0]):
    mask = jnp.zeros(array.shape)
    mask = mask.at[: idx + 1].set(-jnp.inf)
    masks = masks.at[idx].set(mask)

  def unique_mask_locations_counts_it(unique_mask_locations_counts, it):
    # In unique_mask:
    # -1: not seen nor yet a copy
    # 0: already a copy
    # 1: unique element

    # In location:
    # -1: not seen nor yet a copy
    # i: copy of the element at location i

    # In counts:
    # -1: not seen nor yet a copy
    # 0: already a copy
    # i: unique elements with i copies

    unique_mask, locations, counts = unique_mask_locations_counts
    not_seen_yet = abs(unique_mask[it])

    row = jax.lax.dynamic_slice(array, (it, 0), (1,) + array.shape[1:])
    this_mask = jax.lax.dynamic_slice(masks, (it, 0, 0), (1,) + array.shape)
    array_masked = (array + this_mask)[0]

    is_copy = jnp.all(row - array_masked == 0, axis=1).astype(float)

    # 0s or above are left unchanged
    new_unique_mask = (is_copy - 1) * (unique_mask < 0) + unique_mask * (
        unique_mask >= 0
    )
    new_unique_mask = new_unique_mask.at[it].set(not_seen_yet)

    new_locations = ((it + 1) * is_copy - 1.0) * (locations < 0) + locations * (
        locations >= 0
    )
    new_locations = new_locations.at[it].set(
        not_seen_yet * it + (1 - not_seen_yet) * locations[it]
    )

    new_counts = (is_copy - 1) * (counts < 0) + counts * (counts >= 0)
    new_counts = new_counts.at[it].set(not_seen_yet * (jnp.sum(is_copy) + 1))
    return (new_unique_mask, new_locations, new_counts), None

  unique_mask = -jnp.ones(n_rows)
  locations = -jnp.ones(n_rows)
  counts = -jnp.ones(n_rows)
  unique_mask, locations, counts = jax.lax.scan(
      unique_mask_locations_counts_it,
      (unique_mask, locations, counts),
      jnp.arange(n_rows),
  )[0]
  return unique_mask, locations, counts


# def test():
#   arr = np.array(
#       [[1, 2], [1, 2], [2, 3], [2, 4], [2, 3], [2, 5], [2, 5], [1, 2]]
#   )
#   get_unique_mask_location(arr)
#   # np.unique(arr, axis=0, return_index=True, return_inverse=True)
#   pass


########################################################
###################### Init utils ######################
########################################################


def init_log_potentials(
    log_potentials_shape,
    proba_init,
    leak_potentials_mask,
    leak_proba_init,
    dont_update_potentials_mask,
    leak_proba_init_not_updated,
    noise_temperature_init,
    min_clip,
):
  """Initialize the array of log potentials."""
  # First define the probabilities
  proba_init = np.full(log_potentials_shape, fill_value=proba_init)

  # Add noise to break symmetry
  proba_init += noise_temperature_init * np.random.uniform(
      low=-1.0, high=1.0, size=log_potentials_shape
  )

  # Optionally initialize the edges to leak differently
  if leak_potentials_mask is not None:
    leak_proba_init = np.full(log_potentials_shape, fill_value=leak_proba_init)
    proba_init += leak_potentials_mask * (leak_proba_init - proba_init)

  # Optionally initialize some fixed edges differently
  if dont_update_potentials_mask is not None:
    leak_proba_init_not_updated = np.full(
        log_potentials_shape, fill_value=leak_proba_init_not_updated
    )
    proba_init += dont_update_potentials_mask * (
        leak_proba_init_not_updated - proba_init
    )

  # Clip the probabilities
  proba_init = jnp.clip(proba_init, 0.0, 1.0)

  # Define the log potentials
  log_potentials = jnp.full(
      log_potentials_shape, fill_value=-jnp.log(proba_init)
  )

  # Clip the log potentials
  log_potentials = jnp.clip(log_potentials, min_clip, None)
  return log_potentials


########################################################
###################### VI utils ########################
########################################################


def get_value_by_indices(arr, indices, has_multidim_arrays):
  """Returns the values associated to indices, or arrays of indices."""
  assert isinstance(arr, jnp.ndarray)

  if has_multidim_arrays:
    if not isinstance(indices, jnp.ndarray):
      raise TypeError(
          f"Expected indices of type tuple or jax array. Got {type(indices)}"
      )

    if indices.shape[0] == 0:
      return 0.0
    elif indices.ndim == 1:
      return arr[tuple(indices)]
    else:
      return jax.vmap(lambda idx: arr[tuple(idx)], in_axes=0)(indices)

  else:
    # Fill out of bounds value
    return arr.at[indices].get(mode="fill", fill_value=0.0)


def set_value_for_indices(arr, indices, values, has_multidim_arrays):
  """Set the values associated to indices, or arrays of indices."""
  assert isinstance(arr, jnp.ndarray)

  if has_multidim_arrays:
    if not isinstance(indices, jnp.ndarray):
      raise TypeError(
          f"Expected indices of type tuple or jax array. Got {type(indices)}"
      )

    if indices.shape[0] == 0:
      return arr
    elif indices.ndim == 1:
      # Single update
      return arr.at[tuple(indices)].set(values)
    else:

      def f(arr, it):
        """Useful function."""
        idx = indices[it]
        val = values[it]
        return arr.at[tuple(idx)].set(val), None

      return jax.lax.scan(f, arr, jnp.arange(values.shape[0]))[0]

  else:
    # Drop out of bounds indices
    return arr.at[indices].set(values, mode="promise_in_bounds")


def add_value_to_indices(arr, indices, values, has_multidim_arrays):
  """Set the values associated to indices, or arrays of indices."""
  assert isinstance(arr, jnp.ndarray)

  if has_multidim_arrays:
    if not isinstance(indices, jnp.ndarray):
      raise TypeError(
          f"Expected indices of type tuple or jax array. Got {type(indices)}"
      )

    if indices.shape[0] == 0:
      return arr
    elif indices.ndim == 1:
      return arr.at[tuple(indices)].add(values)
    else:

      def f(arr, it):
        """Useful function."""
        idx = indices[it]
        val = values[it]
        return arr.at[tuple(idx)].add(val), None

      return jax.lax.scan(f, arr, jnp.arange(values.shape[0]))[0]

  else:
    # Drop out of bounds indices
    return arr.at[indices].add(values, mode="promise_in_bounds")


def build_local_model(Xv_gt, dict_child_to_parents, n_layers):
  """Build local models as described in the VI paper, Section 5.1."""
  # Build local models of active hidden variables
  Xh_gt = []
  for Xv_row in Xv_gt:
    Xh_row = []
    Xh_row_layer = Xv_row

    # At each layer, extract all the parents from the layer above
    for _ in range(n_layers):
      if len(Xh_row_layer) == 0:
        break
      Xh_row_next_layer = np.concatenate(
          [dict_child_to_parents[idx_child] for idx_child in Xh_row_layer]
      )
      Xh_row_next_layer = np.unique(Xh_row_next_layer).tolist()
      Xh_row += Xh_row_next_layer
      # Update
      Xh_row_layer = Xh_row_next_layer
    assert len(Xh_row_layer) == 0
    Xh_gt.append(Xh_row)
  return Xh_gt


def dict_to_array(d, is_multidim, dtype, fill_value):
  """Convert a dict to an array with padding. Keys can be tuple."""
  keys = list(d.keys())
  max_n_values = max([len(v) for v in d.values()])

  if not is_multidim:
    key_max = max(keys)
    keys_shape = (key_max + 1,)
    assert keys_shape == (len(keys),)
    mat_shape = keys_shape + (max_n_values,)
  else:
    key_maxes = np.max(np.array(keys), axis=0)
    keys_shape = tuple(x + 1 for x in key_maxes)
    n_dim = key_maxes.shape[0]
    mat_shape = keys_shape + (max_n_values,) + (n_dim,)

  # Create matrix with default value, which is being broadcasted
  array = np.full(shape=mat_shape, fill_value=fill_value, dtype=dtype)

  for k, v in d.items():
    if len(v) > 0:
      array[k][: len(v)] = v
  return jnp.array(array)


def list_of_arrays_to_array(list_of_mats, dtype, fill_value):
  """Convert a list of arrays to an array with padding."""
  max_n_cols = max([len(row) for row in list_of_mats])
  array = np.zeros((len(list_of_mats), max_n_cols), dtype=dtype)
  for idx, row in enumerate(list_of_mats):
    array[idx, : len(row)] = row
    array[idx, len(row) :] = fill_value
  return jnp.array(array)
