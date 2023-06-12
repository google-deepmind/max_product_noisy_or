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

"""Reimplements the authors' approach of http://auai.org/uai2019/proceedings/papers/317.pdf in JAX."""

import datetime
import functools
import itertools
import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

from mp_noisy_or import data
from mp_noisy_or import utils


# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=comparison-with-itself
def create_indices_arrays(
    edges_c2p, X_shape, log_potentials_shape, hidden_visible_vars, leak_node_idx
):
  """Create dictionnaries mapping children to parents and parents' potentials, as well as parents to children and children's potentials."""
  dict_child_to_parents = {}
  dict_child_to_parents_potentials = {}
  dict_parent_to_children = {idx: [] for idx in hidden_visible_vars}
  dict_parent_to_children_potentials = {idx: [] for idx in hidden_visible_vars}

  # Also return an array mapping nodes to the index connecting them to the leak
  if len(log_potentials_shape) == 1:
    nodes_to_leak = np.zeros(shape=X_shape)
  else:
    nodes_to_leak = np.zeros(shape=X_shape + (len(log_potentials_shape),))

  for idx_child, idx_parents_idx_potentials in edges_c2p.items():
    idx_parents = []
    idx_parents_potentials = []
    for idx_parent, idx_potential in idx_parents_idx_potentials.items():
      if idx_parent == leak_node_idx:
        # Node to leak
        nodes_to_leak[idx_child] = idx_potential
      else:
        # Parent to child
        dict_parent_to_children[idx_parent].append(idx_child)
        dict_parent_to_children_potentials[idx_parent].append(idx_potential)

        idx_parents.append(idx_parent)
        idx_parents_potentials.append(idx_potential)

    # Node to parents
    dict_child_to_parents[idx_child] = jnp.array(idx_parents, dtype=int)
    dict_child_to_parents_potentials[idx_child] = jnp.array(
        idx_parents_potentials, dtype=int
    )

  # Cast as int
  for idx_parent, idx_children in dict_parent_to_children.items():
    dict_parent_to_children[idx_parent] = jnp.array(idx_children, dtype=int)

  for idx_parent, idx_potentials in dict_parent_to_children_potentials.items():
    dict_parent_to_children_potentials[idx_parent] = jnp.array(
        idx_potentials, dtype=int
    )

  nodes_to_leak = jnp.array(nodes_to_leak, dtype=int)

  return (
      dict_child_to_parents,
      dict_child_to_parents_potentials,
      dict_parent_to_children,
      dict_parent_to_children_potentials,
      nodes_to_leak,
  )


class NoisyOR_VI:
  """Trains a NoisyOR model with VI by reimplementing the authors' approach."""

  def __init__(self, config):
    self.config = config
    np.random.seed(self.config.seed)

    # Load data
    (
        self.Xv_gt_train,
        self.Xv_gt_test,
        self.edges_c2p,
        self.X_shape,
        self.log_potentials_shape,
        self.leak_potentials_mask,
        self.dont_update_potentials_mask,
        self.slice_visible,
        self.slice_hidden,
        self.leak_node_idx,
    ) = data.DATA_LOADER[self.config.data.dataset](**self.config.data.args)

    # If the data is not an array of fixed shape, we use local models
    self.use_local_model = not isinstance(self.Xv_gt_train, np.ndarray)
    self.has_multidim_arrays = len(self.X_shape) > 1
    assert not self.has_multidim_arrays or not self.use_local_model

    # Train-test split
    if self.Xv_gt_test is None:
      self.Xv_gt_train, self.Xv_gt_test = data.train_test_shuffle_split(
          self.Xv_gt_train, self.config.seed, self.config.data.ratio_train
      )
    else:
      np.random.shuffle(self.Xv_gt_train)
      np.random.shuffle(self.Xv_gt_test)

    # Compute the visible variables
    if isinstance(self.slice_visible, slice):
      assert not self.has_multidim_arrays
      visible_start = (
          self.slice_visible.start if self.slice_visible.start else 0
      )
      self.visible_vars = list(range(visible_start, self.slice_visible.stop))
    else:
      visible_starts_stops = []
      for this_slice in self.slice_visible:
        if isinstance(this_slice, int):
          visible_starts_stops.append(range(this_slice, this_slice + 1))
        else:
          visible_start = this_slice.start if this_slice.start else 0
          visible_starts_stops.append(range(visible_start, this_slice.stop))
      self.visible_vars = list(itertools.product(*visible_starts_stops))

    # Compute the hidden variables
    if isinstance(self.slice_hidden, slice):
      assert not self.has_multidim_arrays
      hidden_start = self.slice_hidden.start if self.slice_hidden.start else 0
      self.hidden_vars = list(range(hidden_start, self.slice_hidden.stop))
    else:
      hidden_starts_stops = []
      for this_slice in self.slice_hidden:
        if isinstance(this_slice, int):
          hidden_starts_stops.append(range(this_slice, this_slice + 1))
        else:
          hidden_start = this_slice.start if this_slice.start else 0
          hidden_starts_stops.append(range(hidden_start, this_slice.stop))
      self.hidden_vars = list(itertools.product(*hidden_starts_stops))

    self.hidden_visible_vars = self.visible_vars + self.hidden_vars

    # Extract structure in the form of dicts where values are arrays of indices
    (
        self.dict_child_to_parents,
        self.dict_child_to_parents_potentials,
        self.dict_parent_to_children,
        self.dict_parent_to_children_potentials,
        self.nodes_to_leak,
    ) = create_indices_arrays(
        self.edges_c2p,
        self.X_shape,
        self.log_potentials_shape,
        self.hidden_visible_vars,
        self.leak_node_idx,
    )
    # Fill-in values used for future paddings
    if not self.has_multidim_arrays:
      fill_value_nodes = self.leak_node_idx + 1
      fill_value_potentials = self.log_potentials_shape[0] + 1
    else:
      fill_value_nodes = tuple(x + 1 for x in self.leak_node_idx)
      fill_value_potentials = tuple(x + 1 for x in self.log_potentials_shape)

    # From here, change the structure to only use arrays of fixed shape
    # This will allow jitting the JAX functions
    # For sparse data, build the local models and pad the visible and hidden
    if self.use_local_model:
      # For local models, X_visible and X_hidden are the activations indices
      self.Xh_gt_train = utils.build_local_model(
          self.Xv_gt_train,
          self.dict_child_to_parents,
          self.config.data.args.n_layers,
      )
      self.Xh_gt_test = utils.build_local_model(
          self.Xv_gt_test,
          self.dict_child_to_parents,
          self.config.data.args.n_layers,
      )
      # Pad the visible and hidden variables
      self.Xv_gt_train = utils.list_of_arrays_to_array(
          self.Xv_gt_train, dtype=int, fill_value=fill_value_nodes
      )
      self.Xv_gt_test = utils.list_of_arrays_to_array(
          self.Xv_gt_test, dtype=int, fill_value=fill_value_nodes
      )
      self.Xh_gt_train = utils.list_of_arrays_to_array(
          self.Xh_gt_train, dtype=int, fill_value=fill_value_nodes
      )
      self.Xh_gt_test = utils.list_of_arrays_to_array(
          self.Xh_gt_test, dtype=int, fill_value=fill_value_nodes
      )
    else:
      self.Xh_gt_train = None
      self.Xh_gt_test = None

      self.slice_hidden_visible = np.zeros(shape=self.X_shape, dtype=bool)
      self.slice_hidden_visible[self.slice_hidden] = True
      self.slice_hidden_visible[self.slice_visible] = True

    # Convert to arrays
    self.visible_vars = jnp.array(self.visible_vars)
    self.hidden_vars = jnp.array(self.hidden_vars)
    self.hidden_visible_vars = jnp.array(self.hidden_visible_vars)

    # Convert all the dictionnaries of indices into arrays of fixed shape
    self.arr_child_to_parents = utils.dict_to_array(
        self.dict_child_to_parents,
        self.has_multidim_arrays,
        dtype=int,
        fill_value=fill_value_nodes,
    )
    self.arr_child_to_parents_potentials = utils.dict_to_array(
        self.dict_child_to_parents_potentials,
        self.has_multidim_arrays,
        dtype=int,
        fill_value=fill_value_potentials,
    )
    self.arr_parent_to_children = utils.dict_to_array(
        self.dict_parent_to_children,
        self.has_multidim_arrays,
        dtype=int,
        fill_value=fill_value_nodes,
    )
    self.arr_parent_to_children_potentials = utils.dict_to_array(
        self.dict_parent_to_children_potentials,
        self.has_multidim_arrays,
        dtype=int,
        fill_value=fill_value_potentials,
    )
    print("arr_child_to_parents shape", self.arr_child_to_parents.shape)
    print("arr_parent_to_children shape", self.arr_parent_to_children.shape)
    # For out-of-bounds
    self.nodes_to_leak = self.nodes_to_leak.at[self.leak_node_idx].set(
        fill_value_potentials
    )

    # Create optimizer
    self.opt = optax.adam(learning_rate=config.learning.learning_rate)

  def __hash__(self):
    # pylint: disable=line-too-long
    # See https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
    return hash(tuple(self.edges_c2p.keys()))

  @functools.partial(jax.jit, static_argnames=("self", "is_training"))
  def compute_ELBOs_and_grad(
      self,
      log_potentials,
      Xv_batch,
      Xh_batch=None,
      is_training=True,
  ):
    """Compute the lower-bound of the ELBO and its gradients wrt the log potentials."""
    n_samples = len(Xv_batch)

    # Compute the variational parameter
    if not self.use_local_model:
      all_qs, all_rs = jax.vmap(self.EStep, in_axes=(None, 0))(
          log_potentials, Xv_batch
      )
    else:
      assert Xh_batch is not None
      all_qs, all_rs = jax.vmap(self.EStep, in_axes=(None, 0, 0))(
          log_potentials, Xv_batch, Xh_batch
      )

    all_ELBO_lower_bounds = jax.vmap(
        self.compute_ELBO_lower_bound, in_axes=(None, 0, 0)
    )(log_potentials, all_qs, all_rs)
    chex.assert_equal(all_ELBO_lower_bounds.shape, (n_samples,))

    sum_ELBO_lower_bound = jnp.sum(all_ELBO_lower_bounds)
    avg_ELBO_lower_bound = sum_ELBO_lower_bound / all_ELBO_lower_bounds.shape[0]

    # Compute the gradients of the Elbo wrt the log potentials
    if is_training:
      if not self.use_local_model:
        all_grad_log_potentials = jax.vmap(
            self.compute_grads, in_axes=(None, 0, 0)
        )(log_potentials, all_qs, all_rs)
      else:
        all_grad_log_potentials = jax.vmap(
            self.compute_grads, in_axes=(None, 0, 0, 0, 0)
        )(log_potentials, all_qs, all_rs, Xv_batch, Xh_batch)

      avg_grad_log_potentials = jnp.mean(all_grad_log_potentials, axis=0)
      chex.assert_equal_shape([log_potentials, avg_grad_log_potentials])
      sum_ELBO_mode = None

    else:
      avg_grad_log_potentials = None

      all_ELBOs_mode, _ = jax.vmap(
          self.compute_ELBO_from_samples_given_qs, in_axes=(None, 0)
      )(log_potentials, all_qs)

      chex.assert_equal(all_ELBOs_mode.shape, (n_samples,))
      sum_ELBO_mode = jnp.sum(all_ELBOs_mode)

    return (
        avg_ELBO_lower_bound,
        sum_ELBO_lower_bound,
        avg_grad_log_potentials,
        sum_ELBO_mode,
        all_qs,
    )

  @functools.partial(jax.jit, static_argnames="self")
  def EStep(self, log_potentials, X_visible, X_hidden=None):
    """Fast implementation of the variational expectation step."""
    qs = self.init_q(X_visible, X_hidden)
    rs = self.init_r(log_potentials, X_visible, X_hidden)

    def outer_loop_update(qs_rs, _):
      (qs, rs), _ = jax.lax.scan(
          inner_loop_update, qs_rs, None, self.config.learning.n_inner_loops
      )
      rs = self.update_r(log_potentials, qs, rs, X_visible, X_hidden)
      return (qs, rs), None

    def inner_loop_update(qs_rs, _):
      qs, rs = qs_rs
      qs = self.update_q(log_potentials, qs, rs, X_hidden)
      return (qs, rs), None

    (qs, rs), _ = jax.lax.scan(
        outer_loop_update, (qs, rs), None, self.config.learning.n_outer_loops
    )
    return qs, rs

  @functools.partial(jax.jit, static_argnames="self")
  def init_q(self, X_visible, X_hidden=None):
    """Initialize the qs variables."""
    init_qs = jnp.zeros(shape=self.X_shape)
    if not self.use_local_model:
      init_qs = init_qs.at[self.slice_visible].set(X_visible)
      init_qs = init_qs.at[self.slice_hidden].set(0.5)
      init_qs = init_qs.at[self.leak_node_idx].set(1.0)

    else:
      assert X_hidden is not None
      # For local models, X_visible and X_hidden are the activations indices
      init_qs = init_qs.at[X_visible].set(1.0)
      # Following Section 5.1, we only update the hidden variables in X_hidden
      init_qs = init_qs.at[X_hidden].set(0.5)
      init_qs = init_qs.at[self.leak_node_idx].set(1.0)
    return init_qs

  def set_values(self, args, it):
    """Useful method for setting values sequentially."""
    values, all_new_values, all_idx_potentials = args
    new_values = all_new_values[it]
    idx_potentials = all_idx_potentials[it]

    # Out-of-bounds indices are dropped
    values = utils.set_value_for_indices(
        values, idx_potentials, new_values, self.has_multidim_arrays
    )
    return (values, all_new_values, all_idx_potentials), None

  def add_values(self, args, it):
    """Useful method for adding values sequentially."""
    values, all_new_values, all_idx_potentials = args
    new_values = all_new_values[it]
    idx_potentials = all_idx_potentials[it]

    # Out-of-bounds indices are dropped
    values = utils.add_value_to_indices(
        values, idx_potentials, new_values, self.has_multidim_arrays
    )
    return (values, all_new_values, all_idx_potentials), None

  @functools.partial(jax.jit, static_argnames="self")
  def init_r(self, log_potentials, X_visible, X_hidden=None):
    """Initialize the rs variables, Section 4.1.3.

    Note: Sparse data is represented with static shapes, which allows to scan
    the initialization (similar to dense data).
    """
    # Add epsilon for the division in the definition of u
    init_rs = jnp.zeros(shape=self.log_potentials_shape) + self.config.min_clip

    def init_r_node(init_rs, idx_node, switch):
      """Initialize rs for a single node."""
      idx_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_node,
          self.has_multidim_arrays,
      )
      assert idx_potentials_wo_leak.shape[0] > 0
      potentials_wo_leak = utils.get_value_by_indices(
          log_potentials, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      norm_potentials_wo_leak = potentials_wo_leak / (
          jnp.sum(potentials_wo_leak) + self.config.min_clip
      )
      old_rs = utils.get_value_by_indices(
          init_rs, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      chex.assert_equal_shape([norm_potentials_wo_leak, old_rs])

      # 4.1.3: only set rs for active visible, and for hidden but the leak
      new_rs = switch * norm_potentials_wo_leak + (1 - switch) * old_rs
      return new_rs, idx_potentials_wo_leak

    def init_r_node_visible(init_rs, idx_node):
      """Initialize rs for a single visible node."""
      if not self.use_local_model:
        if self.has_multidim_arrays:
          # Drop the first dimension of visible indices
          return init_r_node(init_rs, idx_node, X_visible[tuple(idx_node)[1:]])
        else:
          return init_r_node(init_rs, idx_node, X_visible[idx_node])
      else:
        # For local models, X_visible and X_hidden are the activations indices
        return init_r_node(init_rs, idx_node, 1.0)

    def init_r_node_hidden(init_rs, idx_node):
      """Initialize rs for a single hidden node."""
      return init_r_node(init_rs, idx_node, 1.0)

    if not self.use_local_model:
      visible_vars = self.visible_vars
      hidden_vars = self.hidden_vars
    else:
      assert X_hidden is not None
      visible_vars = X_visible
      hidden_vars = X_hidden

    # Compute the visible initializations in parallel
    all_new_rs, all_idx_potentials_wo_leak = jax.vmap(
        init_r_node_visible, in_axes=(None, 0)
    )(init_rs, visible_vars)

    # Set the visible initializations sequentially
    (init_rs, _, _), _ = jax.lax.scan(
        self.set_values,
        (init_rs, all_new_rs, all_idx_potentials_wo_leak),
        jnp.arange(visible_vars.shape[0]),
    )

    # Compute the hidden initializations in parallel
    all_new_rs, all_idx_potentials_wo_leak = jax.vmap(
        init_r_node_hidden, in_axes=(None, 0)
    )(init_rs, hidden_vars)

    # Set the hidden initializations sequentially
    (init_rs, _, _), _ = jax.lax.scan(
        self.set_values,
        (init_rs, all_new_rs, all_idx_potentials_wo_leak),
        jnp.arange(hidden_vars.shape[0]),
    )
    return init_rs

  @functools.partial(jax.jit, static_argnames="self")
  def update_q(self, log_potentials, qs, rs, X_hidden=None):
    """Update all the qs variables, Section 4.1.2.

    Note: Sparse data is represented with static shapes, which allows to scan
    the updates (similar to dense data).
    """
    if not self.use_local_model:
      order = self.hidden_vars
    else:
      assert X_hidden is not None
      # Step 2, section 5.1: only update the nodes in the local model
      order = X_hidden

    def update_q_node(qs, idx_hidden):
      """Update the qs variable associated to a node, Equations (11)-(12)."""
      # Get indices
      idx_children_potentials = utils.get_value_by_indices(
          self.arr_parent_to_children_potentials,
          idx_hidden,
          self.has_multidim_arrays,
      )
      idx_children = utils.get_value_by_indices(
          self.arr_parent_to_children, idx_hidden, self.has_multidim_arrays
      )
      idx_parents_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_hidden,
          self.has_multidim_arrays,
      )
      idx_parents_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents, idx_hidden, self.has_multidim_arrays
      )
      idx_node_to_leak = utils.get_value_by_indices(
          self.nodes_to_leak, idx_hidden, self.has_multidim_arrays
      )

      # Leak node
      w_0i = utils.get_value_by_indices(
          log_potentials, idx_node_to_leak, self.has_multidim_arrays
      )
      # Equation (12)
      q_i = w_0i + utils.log1mexp(w_0i)

      # Parents
      w_pi = utils.get_value_by_indices(
          log_potentials,
          idx_parents_potentials_wo_leak,
          self.has_multidim_arrays,
      )
      q_pi = utils.get_value_by_indices(
          qs, idx_parents_wo_leak, self.has_multidim_arrays
      )
      r_pi = utils.get_value_by_indices(
          rs, idx_parents_potentials_wo_leak, self.has_multidim_arrays
      )

      # Clipping as out-ot-bounds entries have 0s value
      r_pi = jnp.clip(r_pi, self.config.min_clip, None)

      u_pi = w_0i + w_pi / r_pi
      q_i += jnp.sum(
          q_pi * (w_pi + r_pi * (utils.log1mexp(u_pi) - utils.log1mexp(w_0i)))
      )

      # Children
      nodes_to_leak_children = utils.get_value_by_indices(
          self.nodes_to_leak, idx_children, self.has_multidim_arrays
      )
      w_0c = utils.get_value_by_indices(
          log_potentials, nodes_to_leak_children, self.has_multidim_arrays
      )
      w_ci = utils.get_value_by_indices(
          log_potentials,
          idx_children_potentials,
          self.has_multidim_arrays,
      )
      q_ci = utils.get_value_by_indices(
          qs, idx_children, self.has_multidim_arrays
      )
      r_ci = utils.get_value_by_indices(
          rs, idx_children_potentials, self.has_multidim_arrays
      )

      # Clipping as out-ot-bounds entries have 0s value
      r_ci = jnp.clip(r_ci, self.config.min_clip, None)

      u_ci = w_0c + w_ci / r_ci

      # Equation (12)
      q_i += jnp.sum(
          q_ci * r_ci * (utils.log1mexp(u_ci) - utils.log1mexp(w_0c))
          - (1 - q_ci) * w_ci
      )

      # Equation (11)
      new_q_i = jnp.where(
          q_i >= 0,
          1.0 / (1.0 + jnp.exp(-q_i)),
          jnp.exp(q_i) / (1.0 + jnp.exp(q_i)),
      )
      new_q_i = jnp.clip(new_q_i, self.config.min_clip, None)

      # Out-of-bounds indices are dropped
      qs = utils.set_value_for_indices(
          qs, idx_hidden, new_q_i, self.has_multidim_arrays
      )
      return qs, None

    # Scan the updates
    qs, _ = jax.lax.scan(update_q_node, qs, order)
    return qs

  @functools.partial(jax.jit, static_argnames="self")
  def update_r(self, log_potentials, qs, rs, X_visible, X_hidden=None):
    """Update the rs variables, Section 4.1.1.

    Note: Sparse data is represented with static shapes, which allows to scan
    the updates (similar to dense data).
    """

    def update_r_node(rs, idx_node, switch):
      """Update the rs variable associated to a node."""
      # Get indices
      idx_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_node,
          self.has_multidim_arrays,
      )
      idx_parents_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents, idx_node, self.has_multidim_arrays
      )
      idx_node_to_leak = utils.get_value_by_indices(
          self.nodes_to_leak, idx_node, self.has_multidim_arrays
      )
      assert idx_parents_wo_leak.shape[0] > 0

      # Get parameters values
      w_0i = utils.get_value_by_indices(
          log_potentials, idx_node_to_leak, self.has_multidim_arrays
      )
      w_pi = utils.get_value_by_indices(
          log_potentials, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      q_pi = utils.get_value_by_indices(
          qs, idx_parents_wo_leak, self.has_multidim_arrays
      )
      r_pi = utils.get_value_by_indices(
          rs, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      old_r_pi = r_pi.copy()

      # Clipping as out-ot-bounds entries have 0s value
      r_pi = jnp.clip(r_pi, self.config.min_clip, None)

      def update_r(r_pi, _):
        """Implement Equation (10)."""
        u_pi = w_0i + w_pi / r_pi
        r_pi = q_pi * (
            r_pi * (utils.log1mexp(u_pi) - utils.log1mexp(w_0i))
            - w_pi * f_prime(u_pi)
        )
        r_pi /= jnp.sum(r_pi) + self.config.min_clip
        r_pi = jnp.clip(r_pi, self.config.min_clip, None)
        return r_pi, None

      r_pi, _ = jax.lax.scan(
          update_r, r_pi, None, self.config.learning.n_inner_loops
      )
      chex.assert_equal_shape([r_pi, old_r_pi])
      new_r_pi = switch * r_pi + (1 - switch) * old_r_pi
      new_r_pi = jnp.clip(new_r_pi, self.config.min_clip, None)
      new_r_pi /= jnp.sum(new_r_pi)
      return new_r_pi, idx_potentials_wo_leak

    def update_r_node_visible(rs, idx_node):
      """Single visible node update."""
      if not self.use_local_model:
        if self.has_multidim_arrays:
          # Drop the first dimension of visible indices
          return update_r_node(rs, idx_node, X_visible[tuple(idx_node)[1:]])
        else:
          return update_r_node(rs, idx_node, X_visible[idx_node])
      else:
        # For local models, X_visible and X_hidden are the activations indices
        return update_r_node(rs, idx_node, 1.0)

    def update_r_node_hidden(rs, idx_node):
      """Single hidden node update."""
      return update_r_node(rs, idx_node, 1.0)

    if not self.use_local_model:
      visible_vars = self.visible_vars
      hidden_vars = self.hidden_vars
    else:
      assert X_hidden is not None
      visible_vars = X_visible
      hidden_vars = X_hidden

    # Update the visible variables in parallel, as mentionned in 4.1.1
    all_new_rs, all_idx_potentials_wo_leak = jax.vmap(
        update_r_node_visible, in_axes=(None, 0)
    )(rs, visible_vars)

    # Set the visible updates sequentially
    (rs, _, _), _ = jax.lax.scan(
        self.set_values,
        (rs, all_new_rs, all_idx_potentials_wo_leak),
        jnp.arange(visible_vars.shape[0]),
    )

    # Update the hidden variables in parallel, as mentionned in 4.1.1
    all_new_rs, all_idx_potentials_wo_leak = jax.vmap(
        update_r_node_hidden, in_axes=(None, 0)
    )(rs, hidden_vars)

    # Set the hidden updates sequentially
    (rs, _, _), _ = jax.lax.scan(
        self.set_values,
        (rs, all_new_rs, all_idx_potentials_wo_leak),
        jnp.arange(hidden_vars.shape[0]),
    )
    return rs

  @functools.partial(jax.jit, static_argnames="self")
  def compute_grads(
      self, log_potentials, qs, rs, X_visible=None, X_hidden=None
  ):
    """Compute the gradients of the Elbo wrt log potentials, Section 4.2.1 and 4.2.2.

    Note: Sparse data is represented with static shapes, which allows to scan
    the computation (similar to dense data).
    """

    def compute_grad_node(idx_node):
      """For each node, compute the partial derivatives for (1) the parameters connecting it to the parents and (2) the parameter connecting it to the leak."""
      # Get indices
      idx_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_node,
          self.has_multidim_arrays,
      )
      idx_parents_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents, idx_node, self.has_multidim_arrays
      )
      idx_node_to_leak = utils.get_value_by_indices(
          self.nodes_to_leak, idx_node, self.has_multidim_arrays
      )

      # Leak node
      w_0i = utils.get_value_by_indices(
          log_potentials, idx_node_to_leak, self.has_multidim_arrays
      )

      if idx_parents_wo_leak.shape[0] == 0:
        sum_pi = 0
      else:
        # Parents
        w_pi = utils.get_value_by_indices(
            log_potentials, idx_potentials_wo_leak, self.has_multidim_arrays
        )
        q_pi = utils.get_value_by_indices(
            qs, idx_parents_wo_leak, self.has_multidim_arrays
        )
        r_pi = utils.get_value_by_indices(
            rs, idx_potentials_wo_leak, self.has_multidim_arrays
        )
        q_i = utils.get_value_by_indices(qs, idx_node, self.has_multidim_arrays)

        # Clipping as out-ot-bounds entries have 0s value
        r_pi = jnp.clip(r_pi, self.config.min_clip, None)

        u_pi = w_0i + w_pi / r_pi

        # Eqs (13)-(14): partial derivative wrt edges from parents to child
        grad_pi = jnp.where(
            u_pi >= 0,
            1.0 / (1.0 - jnp.exp(-u_pi)),
            jnp.exp(u_pi) / (jnp.exp(u_pi) - 1.0),
        )
        grad_pi = q_pi * (q_i * grad_pi - 1)
        chex.assert_equal_shape([grad_pi, w_pi])

        # Useful quantity
        sum_pi = jnp.sum(q_pi * r_pi * (f_prime(u_pi) - f_prime(w_0i)))

      # Equations (15)-(16): partial derivative wrt edge from leak to child
      grad_leak_i = q_i * f_prime(w_0i) - (1.0 - q_i) + q_i * sum_pi

      return grad_pi, idx_potentials_wo_leak, grad_leak_i, idx_node_to_leak

    # Initialze the gradient
    grads_log_potentials = jnp.zeros_like(log_potentials)

    if not self.use_local_model:
      hidden_visible_vars = self.hidden_visible_vars
    else:
      assert X_visible is not None
      assert X_hidden is not None
      assert not self.has_multidim_arrays
      hidden_visible_vars = jnp.concatenate([X_visible, X_hidden])

      # This works as self.arr_child_to_parents is 1D
      all_q_pi = utils.get_value_by_indices(
          qs, self.arr_child_to_parents, self.has_multidim_arrays
      )
      # Equations (13)-(14):
      # partial derivative w.r.t non-leak edge not in local models is -q_k
      grads_log_potentials = utils.set_value_for_indices(
          grads_log_potentials,
          self.arr_child_to_parents_potentials,
          -all_q_pi,
          self.has_multidim_arrays,
      )

      # Equations (15)-(16):
      # partial derivative w.r.t leak edge not in local models is -1
      grads_log_potentials = utils.set_value_for_indices(
          grads_log_potentials,
          self.nodes_to_leak,
          -1.0,
          self.has_multidim_arrays,
      )

    # Compute the gradient in parallel
    (
        all_grad_potentials_wo_leak,
        all_idx_potentials_wo_leak,
        all_grad_potentials_leak,
        all_idx_node_to_leak,
    ) = jax.vmap(compute_grad_node, in_axes=(0,))(hidden_visible_vars)

    # Add or set the updates sequentially
    if self.has_multidim_arrays:
      # Note: a parameter can only be shared across edges in the multidim case
      # Gradient for all the parents-children potentials
      (grads_log_potentials, _, _), _ = jax.lax.scan(
          self.add_values,
          (
              grads_log_potentials,
              all_grad_potentials_wo_leak,
              all_idx_potentials_wo_leak,
          ),
          jnp.arange(hidden_visible_vars.shape[0]),
      )
      # Gradient for all the potentials connecting to the leak
      (grads_log_potentials, _, _), _ = jax.lax.scan(
          self.add_values,
          (
              grads_log_potentials,
              all_grad_potentials_leak,
              all_idx_node_to_leak,
          ),
          jnp.arange(hidden_visible_vars.shape[0]),
      )
    else:
      # Gradient for all the parents-children potentials
      (grads_log_potentials, _, _), _ = jax.lax.scan(
          self.set_values,
          (
              grads_log_potentials,
              all_grad_potentials_wo_leak,
              all_idx_potentials_wo_leak,
          ),
          jnp.arange(hidden_visible_vars.shape[0]),
      )
      # Gradient for all the potentials connecting to the leak
      (grads_log_potentials, _, _), _ = jax.lax.scan(
          self.set_values,
          (
              grads_log_potentials,
              all_grad_potentials_leak,
              all_idx_node_to_leak,
          ),
          jnp.arange(hidden_visible_vars.shape[0]),
      )

    return grads_log_potentials

  def compute_entropy(self, qs):
    """Compute the entropy."""
    q_hidden = qs[self.slice_hidden]  # extract the visible variables
    q_hidden_below = jnp.clip(q_hidden, self.config.min_clip, None)
    one_minus_q_hidden_above = jnp.clip(
        1 - q_hidden, self.config.min_clip, None
    )
    arr = q_hidden * jnp.log(q_hidden_below) + (1 - q_hidden) * jnp.log(
        one_minus_q_hidden_above
    )
    entropy = -jnp.sum(arr)
    return entropy

  @functools.partial(jax.jit, static_argnames="self")
  def compute_ELBO_lower_bound(self, log_potentials, qs, rs):
    """Compute the lower bound of the ELBO, Equation (9), for a sample."""

    def compute_ELogP_lower_bound_node(idx_node):
      """Compute the lower bound for a variable, Equation (8)."""
      # Get indices
      idx_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_node,
          self.has_multidim_arrays,
      )
      idx_parents_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents, idx_node, self.has_multidim_arrays
      )
      idx_node_to_leak = utils.get_value_by_indices(
          self.nodes_to_leak, idx_node, self.has_multidim_arrays
      )

      # Get values
      w_0i = utils.get_value_by_indices(
          log_potentials, idx_node_to_leak, self.has_multidim_arrays
      )
      w_pi = utils.get_value_by_indices(
          log_potentials, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      q_pi = utils.get_value_by_indices(
          qs, idx_parents_wo_leak, self.has_multidim_arrays
      )
      r_pi = utils.get_value_by_indices(
          rs, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      q_i = utils.get_value_by_indices(qs, idx_node, self.has_multidim_arrays)

      # Clipping as out-ot-bounds entries have 0s value
      r_pi = jnp.clip(r_pi, self.config.min_clip, None)

      E_log_p_off = -w_0i - jnp.dot(w_pi, q_pi)

      # Compute the expectation in the second line of Equation (8)
      if idx_parents_wo_leak.shape[0] == 0:
        E_log_p_on = utils.log1mexp(w_0i)
      else:
        u_pi = w_0i + w_pi / r_pi
        E_log_p_on = utils.log1mexp(w_0i) + jnp.sum(
            q_pi * r_pi * (utils.log1mexp(u_pi) - utils.log1mexp(w_0i))
        )

      res = q_i * E_log_p_on + (1 - q_i) * E_log_p_off
      return res

    if not self.has_multidim_arrays:
      # Compute the lower bound at each node but the leak
      ELBOs = jax.vmap(compute_ELogP_lower_bound_node)(self.hidden_visible_vars)
      ELBO = jnp.sum(ELBOs)
    else:
      # When the shapes are 1D we can vectorize the computations
      all_w_0i = log_potentials[self.nodes_to_leak[self.slice_hidden_visible]]
      # Out-of-bounds entries are filled in with 0s
      all_w_pi = utils.get_value_by_indices(
          log_potentials,
          self.arr_child_to_parents_potentials,
          has_multidim_arrays=False,
      )
      all_q_pi = utils.get_value_by_indices(
          qs, self.arr_child_to_parents, has_multidim_arrays=False
      )
      all_w_dot_q = jax.vmap(jnp.dot, in_axes=(0, 0))(all_w_pi, all_q_pi)
      chex.assert_equal(all_w_0i.shape, all_w_dot_q.shape)

      all_r_pi = utils.get_value_by_indices(
          rs, self.arr_child_to_parents_potentials, has_multidim_arrays=False
      )

      # Clipping as out-ot-bounds entries have 0s value
      all_r_pi = jnp.clip(all_r_pi, self.config.min_clip, None)

      E_log_p_off = -all_w_0i - all_w_dot_q

      all_u_pi = all_w_0i[:, None] + all_w_pi / all_r_pi
      E_log_p_on = utils.log1mexp(all_w_0i) + jnp.sum(
          all_q_pi
          * all_r_pi
          * (utils.log1mexp(all_u_pi) - utils.log1mexp(all_w_0i)[:, None]),
          axis=1,
      )
      E_log_p_on = E_log_p_on.reshape(-1)
      chex.assert_equal(E_log_p_off.shape, E_log_p_on.shape)

      all_res = (
          qs[self.slice_hidden_visible] * E_log_p_on
          + (1 - qs[self.slice_hidden_visible]) * E_log_p_off
      )
      ELBO = jnp.sum(all_res)

    # Add the entropy
    ELBO += self.compute_entropy(qs)
    return ELBO

  @functools.partial(jax.jit, static_argnames="self")
  def compute_ELBO_from_samples_given_qs(self, log_potentials, all_qs):
    """Given the posterior qs, estimate the posterior mode then compute the max-product ELBO."""
    # Estimate the posterior mode
    X_sample = jnp.round(all_qs)

    def log_joint_lik_node(X_sample, idx_node):
      """Joint likelihood of the binary hidden and visible, Equation (1)."""
      # Get indices
      idx_potentials_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents_potentials,
          idx_node,
          self.has_multidim_arrays,
      )
      idx_parents_wo_leak = utils.get_value_by_indices(
          self.arr_child_to_parents, idx_node, self.has_multidim_arrays
      )
      idx_node_to_leak = utils.get_value_by_indices(
          self.nodes_to_leak, idx_node, self.has_multidim_arrays
      )

      # Get values
      w_0i = utils.get_value_by_indices(
          log_potentials,
          idx_node_to_leak,
          self.has_multidim_arrays,
      )
      w_pi = utils.get_value_by_indices(
          log_potentials, idx_potentials_wo_leak, self.has_multidim_arrays
      )
      X_pi = utils.get_value_by_indices(
          X_sample, idx_parents_wo_leak, self.has_multidim_arrays
      )
      X_i = utils.get_value_by_indices(
          X_sample, idx_node, self.has_multidim_arrays
      )

      sum_factor = w_0i + jnp.dot(w_pi, X_pi)
      log_joint_lik_node = (1 - X_i) * (-sum_factor) + X_i * jnp.clip(
          jnp.log1p(-jnp.exp(-sum_factor)), utils.CLIP_INF, None
      )
      return log_joint_lik_node

    if not self.has_multidim_arrays:
      # Joint likelihood of the hidden and the visible, Equation (1)."""
      log_joint_liks = jax.vmap(log_joint_lik_node, in_axes=(None, 0))(
          X_sample, self.hidden_visible_vars
      )
      log_joint_lik = jnp.sum(log_joint_liks)
    else:
      # When the shapes are 1D we can vectorize the computations
      all_w_0i = log_potentials[self.nodes_to_leak[self.slice_hidden_visible]]
      # Out-of-bouns entries are filled in with 0s
      all_w_pi = utils.get_value_by_indices(
          log_potentials,
          self.arr_child_to_parents_potentials,
          has_multidim_arrays=False,
      )
      X_sample_parents = utils.get_value_by_indices(
          X_sample, self.arr_child_to_parents, has_multidim_arrays=False
      )
      all_w_dot_X = jax.vmap(jnp.dot, in_axes=(0, 0))(
          all_w_pi, X_sample_parents
      )
      chex.assert_equal(all_w_0i.shape, all_w_dot_X.shape)
      sum_factor_XW = all_w_0i + all_w_dot_X

      # Clipping to avoid nan
      # Note: if sum_factor_XW[k] is so small that we need clipping,
      # then X_sample[k] = 0 and we do not care about the second term
      log_p_factors = -sum_factor_XW * (
          1 - X_sample[self.slice_hidden_visible]
      ) + X_sample[self.slice_hidden_visible] * jnp.clip(
          jnp.log1p(-jnp.exp(-sum_factor_XW)), utils.CLIP_INF, None
      )
      log_joint_lik = log_p_factors.sum()
    return log_joint_lik, X_sample

  def update_log_potentials(
      self, Xv_batch, log_potentials, opt_state, Xh_batch=None
  ):
    """Update the log potentials."""
    # Get the loss and the gradients
    (avg_ELBO_lower_bound, _, grad_log_potentials, _, _) = (
        self.compute_ELBOs_and_grad(
            log_potentials, Xv_batch, Xh_batch, is_training=True
        )
    )

    # Update the log potentials
    grad_log_potentials *= -1.0
    updates, new_opt_state = self.opt.update(grad_log_potentials, opt_state)
    new_log_potentials = optax.apply_updates(log_potentials, updates)

    new_log_potentials = jnp.clip(
        new_log_potentials,
        self.config.min_clip,
        None,
    )

    if self.dont_update_potentials_mask is not None:
      new_log_potentials += self.dont_update_potentials_mask * (
          log_potentials - new_log_potentials
      )
    return new_log_potentials, new_opt_state, avg_ELBO_lower_bound

  def eval_ELBOs_dataset(self, Xv, log_potentials, Xh=None):
    """Compute two ELBOs on an entire dataset.

    (1) The first one is the regular ELBO for the VI mean-field posterior
    (2) The second one defines the posterior via a Dirac at its mode
    """
    test_batch_size = self.config.inference.test_batch_size
    n_batches = (len(Xv) + test_batch_size - 1) // test_batch_size

    if Xh is not None:
      assert len(Xv) == len(Xh)

    sum_elbo_lower_bound = 0.0
    sum_elbo_mode = 0.0
    all_qs_batch = []

    for batch_idx in tqdm.trange(n_batches):
      Xv_batch = Xv[
          batch_idx * test_batch_size : (batch_idx + 1) * test_batch_size
      ]
      if self.use_local_model:
        Xh_batch = Xh[
            batch_idx * test_batch_size : (batch_idx + 1) * test_batch_size
        ]
      else:
        Xh_batch = None

      (
          _,
          sum_elbo_lower_bound_batch,
          _,
          sum_elbo_mode_batch,
          qs_batch,
      ) = self.compute_ELBOs_and_grad(
          log_potentials,
          Xv_batch,
          Xh_batch=Xh_batch,
          is_training=False,
      )
      sum_elbo_lower_bound += sum_elbo_lower_bound_batch
      sum_elbo_mode += sum_elbo_mode_batch
      all_qs_batch.append(qs_batch)

    all_qs = np.concatenate(all_qs_batch, axis=0)
    return (sum_elbo_lower_bound / len(Xv), sum_elbo_mode / len(Xv), all_qs)

  def train(self, init_log_potentials=None):
    """Train the noisy OR model with VI."""
    if init_log_potentials is not None:
      log_potentials = init_log_potentials
      log_potentials = jnp.clip(log_potentials, self.config.min_clip, None)
    else:
      log_potentials = utils.init_log_potentials(
          self.log_potentials_shape,
          self.config.learning.proba_init,
          self.leak_potentials_mask,
          self.config.learning.leak_proba_init,
          self.dont_update_potentials_mask,
          self.config.learning.leak_proba_init_not_updated,
          self.config.learning.noise_temperature_init,
          self.config.min_clip,
      )

    opt_state = self.opt.init(log_potentials)
    current_step = 0
    n_steps = self.config.learning.num_iters

    train_batch_size = self.config.learning.train_batch_size
    n_batches = (
        len(self.Xv_gt_train) + train_batch_size - 1
    ) // train_batch_size

    all_update_times = []
    all_eval_times = []
    all_train_avg_elbos_lb = []
    all_test_avg_elbos_mode = []
    all_test_avg_elbos_lb = []

    # Training iterations
    print(f"Training for {n_steps} steps")
    pbar = tqdm.tqdm(range(n_steps + 1))
    display = {}
    for it in pbar:
      # Extract batch
      batch_idx = it % n_batches
      Xv_batch = self.Xv_gt_train[
          batch_idx * train_batch_size : (batch_idx + 1) * train_batch_size
      ]
      if self.use_local_model:
        Xh_batch = self.Xh_gt_train[
            batch_idx * train_batch_size : (batch_idx + 1) * train_batch_size
        ]
      else:
        Xh_batch = None

      # Gradient step
      start_update = time.time()
      (
          log_potentials,
          opt_state,
          avg_elbo_lower_bound,
      ) = self.update_log_potentials(
          Xv_batch, log_potentials, opt_state, Xh_batch=Xh_batch
      )
      train_avg_elbo_lower_bound = float(jax.device_get(avg_elbo_lower_bound))
      display["train_elbo_lb"] = round(train_avg_elbo_lower_bound, 4)

      # First iteration compiles
      if current_step > 0:
        update_time = time.time() - start_update
        all_update_times.append(update_time)
        all_train_avg_elbos_lb.append(train_avg_elbo_lower_bound)

      # Evaluation step
      if (
          current_step % self.config.learning.eval_every == 0
          or current_step == n_steps
      ):
        start_eval = time.time()
        (
            test_avg_elbo_lower_bound,
            test_avg_elbo_mode,
            test_qs,
        ) = self.eval_ELBOs_dataset(
            self.Xv_gt_test[: self.config.inference.test_size_eval_and_store],
            log_potentials,
            self.Xh_gt_test,
        )
        test_avg_elbo_lower_bound = float(
            jax.device_get(test_avg_elbo_lower_bound)
        )
        test_avg_elbo_mode = float(jax.device_get(test_avg_elbo_mode))
        display["test_elbo_lb"] = round(test_avg_elbo_lower_bound, 4)

        eval_time = time.time() - start_eval
        if current_step > 0:
          all_test_avg_elbos_lb.append(test_avg_elbo_lower_bound)
          all_test_avg_elbos_mode.append(test_avg_elbo_mode)
          all_eval_times.append(eval_time)

      # When we store_inference_results, evaluate on the training set in the end
      if (
          current_step == n_steps
          and self.config.inference.store_inference_results
      ):
        (
            last_train_avg_elbo_lower_bound,
            last_train_avg_elbo_mode,
            last_train_qs,
        ) = self.eval_ELBOs_dataset(
            self.Xv_gt_train[: self.config.inference.test_size_eval_and_store],
            log_potentials,
            self.Xh_gt_train,
        )
        last_train_avg_elbo_lower_bound = float(
            jax.device_get(last_train_avg_elbo_lower_bound)
        )
        last_train_avg_elbo_mode = float(
            jax.device_get(last_train_avg_elbo_mode)
        )

      pbar.set_postfix(display)
      current_step += 1

    print("Training finished")
    results = {
        "config": self.config,
        "log_potentials": log_potentials,
        "all_train_avg_elbos_lb": all_train_avg_elbos_lb,
        "all_test_avg_elbos_mode": all_test_avg_elbos_mode,
        "all_test_avg_elbos_lb": all_test_avg_elbos_lb,
        "all_update_times": all_update_times,
        "all_eval_times": all_eval_times,
    }
    return results


def f_prime(x):
  """Stable implementation of the derivative of f(x)=log(1 - exp(-x))."""
  stable_f_prime = jnp.where(
      x >= 0, jnp.exp(-x) / (1.0 - jnp.exp(-x)), 1.0 / (jnp.exp(x) - 1.0)
  )
  return jnp.clip(stable_f_prime, utils.CLIP_INF, None)
