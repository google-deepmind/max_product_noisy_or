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

"""Trains a noisy OR Bayesian network with max-product Belief Propagation in JAX."""

import datetime
import functools
import time

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
from pgmax import factor
from pgmax import fgraph
from pgmax import fgroup
from pgmax import infer
from pgmax import vgroup
import tqdm

from mp_noisy_or import data
from mp_noisy_or import utils


# pylint: disable=g-complex-comprehension
# pylint: disable=g-doc-args
# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield
# pylint: disable=comparison-with-itself
def build_noisy_or_fg(edges_children_to_parents, X_shape):
  """Build a factor graph with the PGMax representation of the noisy OR factors.

  Args:
    edges_children_to_parents: Dict {idx_child: {idx_parent: idx_potential}}
    X_shape: Shape of array containing all the hidden and visible variables

  Returns:
    fg: factor graph with the noisyOR factors
    factor_children_indices: Indices of all children
    factor_parents_indices: Indices of all children
    fator_log_potentials_indices: Indices of log-potentials
  """
  children_to_parents = tuple(
      [
          (idx_child, idx_parent)
          for idx_child, idx_parents_idx_potentials in edges_children_to_parents.items()
          for idx_parent in idx_parents_idx_potentials
      ]
  )
  num_ORFactors = len(children_to_parents)
  children_to_parents_dict = dict(
      zip(children_to_parents, np.arange(num_ORFactors))
  )

  # Define variables groups
  X = vgroup.NDVarArray(num_states=2, shape=X_shape)
  noisy_X = vgroup.NDVarArray(num_states=2, shape=(num_ORFactors,))

  # Define the factor graph
  fg = fgraph.FactorGraph(variable_groups=[X, noisy_X])

  # Loop through the entries of the dictionnary
  variables_for_ORFactors = []
  variables_for_NoisyFactors = []
  factors_log_potentials_indices = []

  for (
      idx_child,
      idx_parents_idx_potentials,
  ) in edges_children_to_parents.items():
    variables_for_ORFactor = []
    for idx_parent, idx_potential in idx_parents_idx_potentials.items():
      parent = X[idx_parent]
      idx_noisy_X = children_to_parents_dict[(idx_child, idx_parent)]
      noisy_parent = noisy_X[idx_noisy_X]
      variables_for_ORFactor.append(noisy_parent)

      # Noisy factor: order matters!
      variables_for_NoisyFactors.append([parent, noisy_parent])
      factors_log_potentials_indices.append(idx_potential)

    child = X[idx_child]
    variables_for_ORFactor.append(child)
    variables_for_ORFactors.append(variables_for_ORFactor)

  # Build the FactorGroups
  OR_factor_group = fgroup.ORFactorGroup(variables_for_ORFactors)

  log_potential_matrix = np.zeros((num_ORFactors,) + (2, 2))
  noisy_factor_group = fgroup.PairwiseFactorGroup(
      variables_for_factors=variables_for_NoisyFactors,
      log_potential_matrix=log_potential_matrix,
  )
  # Add the FactorGroups, which is computationally efficient
  fg.add_factors([OR_factor_group, noisy_factor_group])

  # Factor indices
  factor_children_indices = jnp.array(children_to_parents)[:, 0]
  factor_parents_indices = jnp.array(children_to_parents)[:, 1]
  fator_log_potentials_indices = jnp.array(factors_log_potentials_indices)

  def _get_indices(array):
    """Useful function to process indices."""
    array_by_dim = tuple(
        [array[:, idx_col] for idx_col in range(array.shape[1])]
    )
    return array_by_dim

  if len(X_shape) > 1:
    factor_children_indices = _get_indices(factor_children_indices)
    factor_parents_indices = _get_indices(factor_parents_indices)
    fator_log_potentials_indices = _get_indices(fator_log_potentials_indices)

  return (
      fg,
      factor_children_indices,
      factor_parents_indices,
      fator_log_potentials_indices,
  )


class NoisyOR_BP:
  """Trains a NoisyOR model with max-product."""

  def __init__(self, config):
    self.config = config
    np.random.seed(self.config.seed)
    self.rng = jax.random.PRNGKey(self.config.seed)

    # Load data
    (
        self.Xv_gt_train,
        self.Xv_gt_test,
        self.edges_children_to_parents,
        self.X_shape,
        self.log_potentials_shape,
        self.leak_potentials_mask,
        self.dont_update_potentials_mask,
        self.slice_visible,
        self.slice_hidden,
        self.leak_node_idx,
    ) = data.DATA_LOADER[self.config.data.dataset](**self.config.data.args)

    if self.Xv_gt_test is None:
      # Train-test split
      self.Xv_gt_train, self.Xv_gt_test = data.train_test_shuffle_split(
          self.Xv_gt_train, self.config.seed, self.config.data.ratio_train
      )
    else:
      np.random.shuffle(self.Xv_gt_train)
      np.random.shuffle(self.Xv_gt_test)

    self.has_dense_Xv = isinstance(self.Xv_gt_train, np.ndarray)

    if not isinstance(self.slice_visible, tuple):
      self.slice_visible = (self.slice_visible,)

    if not isinstance(self.slice_hidden, tuple):
      self.slice_hidden = (self.slice_hidden,)

    if not isinstance(self.leak_node_idx, tuple):
      self.leak_node_idx = (self.leak_node_idx,)

    # The mask indicates the hidden and visible variables
    self.X_mask = np.zeros(shape=self.X_shape, dtype=float)
    self.X_mask[self.slice_visible] = 1.0
    self.X_mask[self.slice_hidden] = 1.0

    # Create the factor graph
    (
        self.fg,
        self.factor_children_indices,
        self.factor_parents_indices,
        self.factor_log_potentials_indices,
    ) = build_noisy_or_fg(self.edges_children_to_parents, self.X_shape)
    print("Factor graph created")

    # Get variable group and factor group from the factor graph
    self.X = self.fg.variable_groups[0]
    self.noisy_factor_group = self.fg.factor_groups[factor.EnumFactor][0]

    # Create the BP functions for max-product
    self.bp = infer.build_inferer(self.fg.bp_state, backend=config.backend)

    # Create the optimizer
    self.opt = optax.adam(learning_rate=config.learning.learning_rate)

  def __hash__(self):
    # pylint: disable=line-too-long
    # See https://jax.readthedocs.io/en/latest/faq.html#strategy-2-marking-self-as-static
    return hash(tuple(self.edges_children_to_parents.keys()))

  def densify(self, Xv_batch):
    """Densify a sparse batch of activations."""
    # Get dense matrix of observations
    if self.has_dense_Xv:
      Xv_batch_dense = Xv_batch
    else:
      assert len(self.slice_visible) == 1
      Xv_batch_dense = np.zeros(((len(Xv_batch), self.slice_visible[0].stop)))
      for idx_row, Xv_row in enumerate(Xv_batch):
        if Xv_row.shape[0] > 0:
          Xv_batch_dense[idx_row, Xv_row] = 1
    return Xv_batch_dense

  @functools.partial(
      jax.jit,
      static_argnames=("self", "noise_temperature", "n_hidden_by_sample"),
  )
  def posterior_sample(
      self,
      Xv_batch,
      rng,
      log_potentials,
      noise_temperature,
      n_hidden_by_sample,
  ):
    """Given a batch of visible variables, get samples or the mode of the posterior."""
    n_samples = len(Xv_batch)
    # First, create copies of the visible variables
    if n_hidden_by_sample > 1:
      Xv_batch = jnp.repeat(Xv_batch, repeats=n_hidden_by_sample, axis=0)

    # Second, create the evidence array by clamping the visible variables
    uX = jnp.zeros((n_samples * n_hidden_by_sample,) + self.X_shape + (2,))
    uX = uX.at[(slice(None, None, None),) + self.slice_visible + (0,)].set(
        (2 * Xv_batch - 1) * utils.CLIP_INF
    )
    # utils.CLIP_INF acts as a noisy channel between the observations and X

    # Also clamp the node
    uX = uX.at[(slice(None, None, None),) + self.leak_node_idx + (0,)].set(
        utils.CLIP_INF
    )
    # Third, add Gumbel noise to the hidden variables
    rng, rng_input = jax.random.split(rng)
    # Note: we use a prior of 0.5 for the hidden variables
    hidden_evidence = noise_temperature * jax.random.gumbel(
        rng_input,
        shape=uX[(slice(None, None, None),) + self.slice_hidden].shape,
    )

    uX = uX.at[(slice(None, None, None),) + self.slice_hidden].set(
        hidden_evidence
    )

    # Update the log potentials
    log_potentials_copied = log_potentials[self.factor_log_potentials_indices]
    log_potential_matrix = jnp.zeros((log_potentials_copied.shape[0], 2, 2))
    log_potential_matrix = log_potential_matrix.at[:, 0, 0].set(0.0)
    log_potential_matrix = log_potential_matrix.at[:, 0, 1].set(utils.CLIP_INF)
    log_potential_matrix = log_potential_matrix.at[:, 1, 0].set(
        -log_potentials_copied
    )
    log_potential_matrix = log_potential_matrix.at[:, 1, 1].set(
        utils.log1mexp(log_potentials_copied)
    )

    # Useful function for jax.vmap
    def init_bp_arrays(uX, log_potential_matrix):
      bp_arrays = self.bp.init(
          evidence_updates={self.X: uX},
          log_potentials_updates={
              self.noisy_factor_group: log_potential_matrix
          },
      )
      return bp_arrays

    # Run max-product and get the beliefs
    bp_arrays = jax.vmap(init_bp_arrays, in_axes=(0, None), out_axes=0)(
        uX, log_potential_matrix
    )

    assert self.config.backend == "bp"
    bp_arrays = jax.vmap(
        functools.partial(
            self.bp.run,
            num_iters=self.config.bp.num_iters,
            damping=self.config.bp.damping,
            temperature=self.config.bp.temperature,
        ),
        in_axes=0,
        out_axes=0,
    )(bp_arrays)

    beliefs = jax.vmap(self.bp.get_beliefs, in_axes=0, out_axes=0)(bp_arrays)
    map_states = infer.decode_map_states(beliefs)
    X_samples = map_states[self.X]

    # Clamp the visible variables and leak node
    X_samples_clamped = X_samples.at[
        (slice(None, None, None),) + self.slice_visible
    ].set(Xv_batch)
    X_samples_clamped = X_samples_clamped.at[
        (slice(None, None, None),) + self.leak_node_idx
    ].set(1.0)

    return X_samples_clamped, rng

  @functools.partial(jax.jit, static_argnames="self")
  def log_joint_lik(self, X_samples, log_potentials):
    """Compute the expectation under the posterior of the log joint likelihood."""
    log_potentials_copied = log_potentials[self.factor_log_potentials_indices]

    def log_joint_lik_sample(X_sample):
      """Joint likelihood of the hidden and the visible."""
      # Compute x_k * w_{k -> i}
      X_sample_parents = X_sample[self.factor_parents_indices]
      XW = X_sample_parents * log_potentials_copied

      # Sum for each factor
      sum_factor_XW = (
          jnp.zeros(shape=self.X_shape).at[self.factor_children_indices].add(XW)
      )
      # Clipping to avoid nan
      log_p_factors = -sum_factor_XW * (1 - X_sample) + X_sample * (
          utils.log1mexp(sum_factor_XW)
      )
      # X_mask removes the leak node, which is never the children of a factor
      log_p_factors *= self.X_mask
      log_joint_lik = log_p_factors.sum()
      return log_joint_lik

    log_joint_lik = jax.vmap(log_joint_lik_sample, in_axes=0)(X_samples)
    log_joint_lik = log_joint_lik.sum()
    avg_log_joint_lik = log_joint_lik / X_samples.shape[0]
    return avg_log_joint_lik, log_joint_lik

  @functools.partial(jax.jit, static_argnames=("self", "n_hidden_by_sample"))
  def compute_ELBO_from_samples(
      self, X_samples, log_potentials, n_hidden_by_sample
  ):
    """Compute the ELBO given the posteriors samples of a batch.

    Note: if we use multiple samples, we add the entropy of the posterior here,
    and observe that its gradient vanishes.
    """
    if n_hidden_by_sample == 1:
      return self.log_joint_lik(X_samples, log_potentials)

    def compute_ELBO_from_samples_same_Xv(X_samples_same_Xv):
      """Compute the ELBO given multiple samples for the same posterior."""
      unique_mask, _, counts = utils.get_unique_masks_locations_counts(
          X_samples_same_Xv
      )
      probas = counts / jnp.sum(counts)
      probas = jnp.clip(probas, self.config.min_clip, None)
      entropy = -jnp.sum(unique_mask * probas * jnp.log(probas))

      avg_log_joint_lik, _ = self.log_joint_lik(
          X_samples_same_Xv, log_potentials
      )
      return avg_log_joint_lik + entropy

    # Group the posterior samples with same visible observation
    X_samples_reshaped = X_samples.reshape(
        (-1, n_hidden_by_sample) + X_samples.shape[1:]
    )

    # Compute the Elbo for each observation
    elbo_samples = jax.vmap(
        compute_ELBO_from_samples_same_Xv, in_axes=0, out_axes=0
    )(X_samples_reshaped)

    # Sum the Elbos
    sum_elbo = elbo_samples.sum()
    return sum_elbo / elbo_samples.shape[0], sum_elbo

  @functools.partial(jax.jit, static_argnames="self")
  def compute_gradients(self, X_samples, log_potentials):
    """Compute the gradients of the Elbo in closed-form."""
    log_potentials_copied = log_potentials[self.factor_log_potentials_indices]

    def compute_gradients_sample(X_sample):
      """Compute the gradient for a sample."""
      # Compute x_k * w_{k -> i}
      X_sample_parents = X_sample[self.factor_parents_indices]
      XW = X_sample_parents * log_potentials_copied

      # Sum for each factor
      sum_factor_XW = (
          jnp.zeros(shape=self.X_shape).at[self.factor_children_indices].add(XW)
      )

      # Children for each factor
      X_sample_children = X_sample[self.factor_children_indices]
      sum_factor_XW_children = sum_factor_XW[self.factor_children_indices]

      grad_sample_flat = (
          X_sample_children * X_sample_parents * g(sum_factor_XW_children)
          - X_sample_parents
      )

      # Unflatten the gradients
      grad_sample = (
          jnp.zeros(shape=self.log_potentials_shape)
          .at[self.factor_log_potentials_indices]
          .add(grad_sample_flat)
      )
      return grad_sample

    grad_samples = jax.vmap(compute_gradients_sample, in_axes=0)(X_samples)
    return jnp.mean(grad_samples, axis=0)

  def update_log_potentials(
      self,
      Xv_batch,
      log_potentials,
      opt_state,
      noise_temperature,
      n_hidden_by_sample,
  ):
    """Update the log potentials."""
    # Sample from the posterior
    X_samples, self.rng = self.posterior_sample(
        Xv_batch,
        self.rng,
        log_potentials,
        noise_temperature,
        n_hidden_by_sample,
    )
    # Get the loss and the gradients
    avg_elbo, _ = self.compute_ELBO_from_samples(
        X_samples, log_potentials, n_hidden_by_sample
    )
    grad_log_potentials = self.compute_gradients(X_samples, log_potentials)
    chex.assert_equal_shape([log_potentials, grad_log_potentials])

    # Update the log potentials
    updates, new_opt_state = self.opt.update(-grad_log_potentials, opt_state)
    new_log_potentials = optax.apply_updates(log_potentials, updates)

    new_log_potentials = jnp.clip(
        new_log_potentials,
        self.config.min_clip,
        None,
    )

    # Do not update the fixed potentials
    if self.dont_update_potentials_mask is not None:
      new_log_potentials += self.dont_update_potentials_mask * (
          log_potentials - new_log_potentials
      )
    return new_log_potentials, new_opt_state, avg_elbo

  def eval_ELBOs_dataset(
      self,
      Xv,
      log_potentials,
      noise_temperature,
      n_hidden_by_sample,
      test_batch_size,
  ):
    """Compute the Elbo on an entire dataset."""
    n_batches = (len(Xv) + test_batch_size - 1) // test_batch_size

    all_sum_elbo_samples = 0.0
    all_X_samples = []
    for batch_idx in range(n_batches):
      Xv_batch = Xv[
          batch_idx * test_batch_size : (batch_idx + 1) * test_batch_size
      ]
      Xv_batch_dense = self.densify(Xv_batch)

      # Get the mode or a sample from the posterior
      X_samples, self.rng = self.posterior_sample(
          Xv_batch_dense,
          self.rng,
          log_potentials,
          noise_temperature,
          n_hidden_by_sample,
      )
      all_X_samples.append(X_samples)

      # Compute the Elbo
      _, sum_elbo_samples = self.compute_ELBO_from_samples(
          X_samples, log_potentials, n_hidden_by_sample
      )
      all_sum_elbo_samples += sum_elbo_samples

    X_samples = np.concatenate(all_X_samples, axis=0)
    return all_sum_elbo_samples / len(Xv), X_samples

  def train(self):
    """Train the noisy OR model."""
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

    train_noise_temperature = self.config.learning.noise_temperature
    train_n_hidden_by_sample = self.config.learning.n_hidden_by_sample
    test_noise_temperature = self.config.inference.noise_temperature
    test_n_hidden_by_sample = self.config.inference.n_hidden_by_sample
    test_batch_size = self.config.inference.test_batch_size
    n_steps = self.config.learning.num_iters

    train_batch_size = self.config.learning.train_batch_size
    n_batches = (
        len(self.Xv_gt_train) + train_batch_size - 1
    ) // train_batch_size

    all_update_times = []
    all_train_avg_elbos = []
    all_eval_times = []
    all_test_avg_elbos_mode = []

    # Training iterations
    print(f"Training for {n_steps} steps")
    pbar = tqdm.tqdm(range(n_steps + 1))
    display = {}
    for it in pbar:
      batch_idx = it % n_batches
      Xv_batch = self.Xv_gt_train[
          batch_idx * train_batch_size : (batch_idx + 1) * train_batch_size
      ]

      # Gradient step
      start_update = time.time()
      Xv_batch_dense = self.densify(Xv_batch)
      (
          log_potentials,
          opt_state,
          train_avg_elbo,
      ) = self.update_log_potentials(
          Xv_batch_dense,
          log_potentials,
          opt_state,
          train_noise_temperature,
          train_n_hidden_by_sample,
      )
      train_avg_elbo = float(jax.device_get(train_avg_elbo))
      display["train_elbo"] = round(train_avg_elbo, 4)

      # First iteration compiles
      if current_step > 0:
        update_time = time.time() - start_update
        all_update_times.append(update_time)
        all_train_avg_elbos.append(train_avg_elbo)

      # Evaluation step
      if (
          current_step % self.config.learning.eval_every == 0
          or current_step == n_steps
      ):
        start_eval = time.time()
        test_avg_elbo_mode, test_X_samples = self.eval_ELBOs_dataset(
            self.Xv_gt_test[: self.config.inference.test_size_eval_and_store],
            log_potentials,
            test_noise_temperature,
            test_n_hidden_by_sample,
            test_batch_size,
        )
        test_avg_elbo_mode = float(jax.device_get(test_avg_elbo_mode))
        display["test_elbo"] = round(test_avg_elbo_mode, 4)

        eval_time = time.time() - start_eval
        if current_step > 0:
          all_eval_times.append(eval_time)
          all_test_avg_elbos_mode.append(test_avg_elbo_mode)

      # When we store_inference_results, evaluate on the training set in the end
      if (
          current_step == n_steps
          and self.config.inference.store_inference_results
      ):
        last_train_avg_elbo_mode, last_train_X_samples = (
            self.eval_ELBOs_dataset(
                self.Xv_gt_train[
                    : self.config.inference.test_size_eval_and_store
                ],
                log_potentials,
                test_noise_temperature,
                test_n_hidden_by_sample,
                test_batch_size,
            )
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
        "all_train_avg_elbos": all_train_avg_elbos,
        "all_test_avg_elbos_mode": all_test_avg_elbos_mode,
        "all_update_times": all_update_times,
        "all_eval_times": all_eval_times,
        "test_X_samples": test_X_samples,
    }
    return results


def g(x):
  """Stable implementation of g(x) = 1 / (1 - exp(-x))."""
  stable_g = jnp.where(
      x >= 0, 1.0 / (1.0 - jnp.exp(-x)), jnp.exp(x) / (jnp.exp(x) - 1.0)
  )
  return jnp.clip(stable_g, utils.CLIP_INF, None)
