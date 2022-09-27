# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library to training NeRFs."""
from ast import Str
import functools
from tokenize import String
from typing import Any, Callable, Dict
import math
import time
from absl import logging
import numpy as np

import flax
from flax import struct
from flax import traverse_util
from flax import jax_utils
from flax.training import checkpoints
import jax
from jax import lax
from jax import numpy as jnp
from jax import random
from jax import vmap
from jax import tree_util

from hypernerf import model_utils
from hypernerf import editing_models_vanilla as models
from hypernerf import utils
from hypernerf import debug
from tqdm import tqdm

import numpy as np

from transformers import CLIPTokenizer, FlaxCLIPModel, CLIPProcessor
from jax.experimental.host_callback import call

@struct.dataclass
class ScalarParams:
  """Scalar parameters for training."""
  learning_rate: float
  background_loss_weight: float = 0.0
  background_noise_std: float = 0.001
  lambda_clip: float = 1.0

def save_checkpoint(path, state, keep=2):
  """Save the state to a checkpoint."""
  state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
  step = state_to_save.optimizer.state.step
  checkpoint_path = checkpoints.save_checkpoint(
      path, state_to_save, step, keep=keep)
  logging.info('Saved checkpoint: step=%d, path=%s', int(step), checkpoint_path)
  return checkpoint_path


def zero_adam_param_states(state: flax.optim.OptimizerState, selector: str):
  """Applies a gradient for a set of parameters.

  Args:
    state: a named tuple containing the state of the optimizer
    selector: a path string defining which parameters to freeze.

  Returns:
    A tuple containing the new parameters and the new optimizer state.
  """
  step = state.step
  params = flax.core.unfreeze(state.param_states)
  flat_params = {'/'.join(k): v
                 for k, v in traverse_util.flatten_dict(params).items()}
  for k in flat_params:
    if k.startswith(selector):
      v = flat_params[k]
      # pylint: disable=protected-access
      flat_params[k] = flax.optim.adam._AdamParamState(
          jnp.zeros_like(v.grad_ema), jnp.zeros_like(v.grad_sq_ema))

  new_param_states = traverse_util.unflatten_dict(
      {tuple(k.split('/')): v for k, v in flat_params.items()})
  new_param_states = dict(flax.core.freeze(new_param_states))
  new_state = flax.optim.OptimizerState(step, new_param_states)
  return new_state


@jax.jit
def nearest_rotation_svd(matrix, eps=1e-6):
  """Computes the nearest rotation using SVD."""
  # TODO(keunhong): Currently this produces NaNs for some reason.
  u, _, vh = jnp.linalg.svd(matrix + eps, compute_uv=True, full_matrices=False)
  # Handle the case when there is a flip.
  # M will be the identity matrix except when det(UV^T) = -1
  # in which case the last diagonal of M will be -1.
  det = jnp.linalg.det(utils.matmul(u, vh))
  m = jnp.stack([jnp.ones_like(det), jnp.ones_like(det), det], axis=-1)
  m = jnp.diag(m)
  r = utils.matmul(u, utils.matmul(m, vh))
  return r

def _clip_img_preprocess(rgb):
  rgb = jax.image.resize(rgb, (224, 224, 3), 'bilinear')
  rgb = rgb[jnp.newaxis, ...]
  rgb = jnp.transpose(rgb, (0, 3, 1, 2))
  mean = jnp.array([0.48145466, 0.4578275, 0.40821073])
  mean = jnp.expand_dims(mean, axis=(0, 2, 3))
  std = jnp.array([0.26862954, 0.26130258, 0.27577711])
  std = jnp.expand_dims(std, axis=(0, 2, 3))
  return (rgb - mean) / std


@jax.jit
def compute_clip_loss(text_feat, rgb, refrgb):
  model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  rgb = _clip_img_preprocess(rgb) 
  refrgb = _clip_img_preprocess(refrgb)
  # Preprocess image suitable for CLIP model 
  rgb_feat = model.get_image_features(rgb)
  refrgb_feat = model.get_image_features(refrgb)
  delta_feat = rgb_feat - refrgb_feat
  delta_feat = delta_feat / jnp.linalg.norm(delta_feat, axis=1)[..., jnp.newaxis]
  sim_mat = jnp.matmul(text_feat, jnp.transpose(delta_feat, (1, 0)))
  return 1. - jnp.mean(sim_mat)



@functools.partial(jax.jit, static_argnums=0)
def compute_background_loss(model, state, params, key, points, noise_std,
                            alpha=-2, scale=0.001):
  """Compute the background regularization loss."""
  metadata = random.choice(key, model.warp_embeds, shape=(points.shape[0], 1))
  point_noise = noise_std * random.normal(key, points.shape)
  points = points + point_noise
  warp_fn = functools.partial(model.apply, method=model.apply_warp)
  warp_fn = jax.vmap(warp_fn, in_axes=(None, 0, 0, None))
  warp_out = warp_fn({'params': params}, points, metadata, state.extra_params)
  warped_points = warp_out['warped_points'][..., :3]
  sq_residual = jnp.sum((warped_points - points)**2, axis=-1)
  loss = utils.general_loss_with_squared_residual(
      sq_residual, alpha=alpha, scale=scale)
  return loss


@functools.partial(jax.jit,
                   static_argnums=0,
                   static_argnames=('disable_hyper_grads',
                                    'grad_max_val',
                                    'grad_max_norm',
                                    'use_elastic_loss',
                                    'use_background_loss',
                                    'use_warp_reg_loss',
                                    'use_hyper_reg_loss',
                                    'chunk_size',
                                    'epsilon',
                                    'use_mapper'))
def train_step(model: models.EditingNerfModel,
               rng_key: Callable[[int], jnp.ndarray],
               state: model_utils.TrainState,
               batch: Dict[str, Any],
               scalar_params: ScalarParams,
               clip_text: jnp.ndarray,
               use_elastic_loss: bool = False,
               use_background_loss: bool = False,
               use_warp_reg_loss: bool = False,
               use_hyper_reg_loss: bool = False,
               chunk_size: int = 4096,
               epsilon: float = 0.15,
               use_mapper: bool = False):
  """One optimization step.

  Args:
    model: the model module to evaluate.
    rng_key: The random number generator.
    state: model_utils.TrainState, state of model and optimizer.
    batch: dict. A mini-batch of data for training.
    scalar_params: scalar-valued parameters.
    disable_hyper_grads: if True disable gradients to the hyper coordinate
      branches.
    grad_max_val: The gradient clipping value (disabled if == 0).
    grad_max_norm: The gradient clipping magnitude (disabled if == 0).
    use_elastic_loss: is True use the elastic regularization loss.
    elastic_reduce_method: which method to use to reduce the samples for the
      elastic loss. 'median' selects the median depth point sample while
      'weight' computes a weighted sum using the density weights.
    elastic_loss_type: which method to use for the elastic loss.
    use_background_loss: if True use the background regularization loss.
    use_warp_reg_loss: if True use the warp regularization loss.
    use_hyper_reg_loss: if True regularize the hyper points.
    epsilon: a hyperparameter determining the editing near-far bound for prior sampling

  Returns:
    new_state: model_utils.TrainState, new training state.
    stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
  """
  rng_key, fine_key, coarse_key, reg_key, sparse_key = random.split(rng_key, 5)
  batch_shape = batch['origins'].shape[:-1]
  num_rays = np.prod(batch_shape)
  batch = tree_util.tree_map(lambda x: x.reshape((num_rays, -1)), batch)

  # pylint: disable=unused-argument
  def _compute_loss_and_stats(model_out, batch_shape):
    loss = 0.0
    stats = {}
    # L-2 Regularization loss
    # if 'channel_set' in batch['metadata']:
    #   num_sets = int(model_out['rgb'].shape[-1] / 3)
    #   losses = []
    #   for i in range(num_sets):
    #     loss = (model_out['rgb'][..., i * 3:(i + 1) * 3] - batch['rgb'])**2
    #     loss *= (batch['metadata']['channel_set'] == i)
    #     losses.append(loss)
    #   rgb_loss = jnp.sum(jnp.asarray(losses), axis=0).mean()
    # else:
    #rgb_loss = ((model_out['rgb'][..., :3] - batch['rgb'][..., :3])**2).mean()
    #stats = {
    #     'loss/rgb': rgb_loss,
    # }
    #loss += 0.5 * rgb_loss
    
    # Clip loss
    rgb_image = jnp.reshape(model_out['rgb'][..., :3], (batch_shape[0], batch_shape[1], -1))
    refrgb_image = jnp.reshape(batch['rgb'][..., :3], (batch_shape[0], batch_shape[1], -1))
    clip_loss = compute_clip_loss(clip_text, rgb_image, refrgb_image)

    #loss += scalar_params.clip_loss_weight * clip_loss
    loss += scalar_params.lambda_clip * clip_loss
    
    stats['loss/clip'] = clip_loss

    # Other stats computation
    stats['loss/total'] = loss
    return loss, stats

  def render_by_chunk(params, p=0.6):
    """
    Rendering an instance of images with gradients of randomly selected patches disabled (to prevent OOM)

    Args:
      params: network params (likely from state.optimizer.target)
      p: probability that the network outputs' gradient's stopped. (increase p if OOM)

    Return:
      ret_maps: retrieved network outputs with some of the chunks' gradients are disabled
    """
    ret_maps = []
    num_batches = int(math.ceil(num_rays / chunk_size))
    stop_grad_fn = lambda x: jax.lax.stop_gradient(x)
    batch_iter = jnp.arange(math.ceil(num_batches))
    shuffled_batch_iter = jax.random.shuffle(sparse_key, batch_iter)
    #shuffled_batch_iter = jax.random.permutation(sparse_key, batch_iter, independent=True)
    nograd_shuffled_batch_iter = shuffled_batch_iter[:math.ceil(p*num_batches)]
    grad_shuffled_batch_iter = shuffled_batch_iter[math.ceil(p*num_batches):]
    rev_idx = jnp.argsort(shuffled_batch_iter)
    def _chunk_process(batch_idx):
      ray_idx = batch_idx * chunk_size
      chunk_slice_fn = lambda x: jax.lax.dynamic_slice_in_dim(x, ray_idx, chunk_size, axis=0)
      chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, batch)
      ret = model.apply({'params': params['model']},
                        chunk_rays_dict,
                        extra_params=state.extra_params,
                        return_points=(use_warp_reg_loss or use_hyper_reg_loss),
                        return_weights=(use_warp_reg_loss or use_elastic_loss),
                        return_warp_jacobian=use_elastic_loss,
                        rngs={
                            'coarse': coarse_key,
                            'fine': fine_key
                        },
                        epsilon=epsilon,
                        use_mapper=use_mapper)
      return ret
    for batch_idx in nograd_shuffled_batch_iter:
      ret = _chunk_process(batch_idx)
      ret = tree_util.tree_map(stop_grad_fn, ret)
      ret_maps.append(ret)

    for batch_idx in grad_shuffled_batch_iter:
      ret = _chunk_process(batch_idx)
      ret_maps.append(ret)

    ret_map = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *ret_maps)
    _, ret_map_pytreedef = jax.tree_util.tree_flatten(ret_map)
    rev_tree = jax.tree_util.tree_unflatten(ret_map_pytreedef, [rev_idx for i in range(len(_))])
    ret_map = jax.tree_util.tree_map(lambda x, rev: x[rev], ret_map, rev_tree)
    ret_map = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], -1)), ret_map)
    return ret_map

  def _loss_fn(params, batch_shape):
    ret = render_by_chunk(params)
    losses = {}
    stats = {}
    losses['main'], stats['main'] = _compute_loss_and_stats(ret['fine'], batch_shape)

    if use_background_loss:
      background_loss = compute_background_loss(
          model,
          state=state,
          params=params['model'],
          key=reg_key,
          points=batch['background_points'],
          noise_std=scalar_params.background_noise_std)
      background_loss = background_loss.mean()
      losses['background'] = (
          scalar_params.background_loss_weight * background_loss)
      stats['background_loss'] = background_loss
    return sum(losses.values()), (stats, ret)

  optimizer = state.optimizer
  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (_, (stats, model_out)), grad = grad_fn(optimizer.target, batch_shape)
  grad = jax.lax.pmean(grad, axis_name='batch')
  stats = jax.lax.pmean(stats, axis_name='batch')
  new_optimizer = optimizer.apply_gradient(
      grad, learning_rate=scalar_params.learning_rate)
  new_state = state.replace(optimizer=new_optimizer)
  model_out = jax.lax.pmean(model_out, axis_name='batch')
  return new_state, stats, rng_key, model_out, grad


def encode_metadata(model, params, metadata):
  """Encodes metadata embeddings.

  Args:
    model: a NerfModel.
    params: the parameters of the model.
    metadata: the metadata dict.

  Returns:
    A new metadata dict with the encoded embeddings.
  """
  encoded_metadata = {}
  if model.use_nerf_embed:
    encoded_metadata['encoded_nerf'] = model.apply(
        {'params': params}, metadata, method=model.encode_nerf_embed)
  if model.use_warp:
    encoded_metadata['encoded_warp'] = model.apply(
        {'params': params}, metadata, method=model.encode_warp_embed)
  if model.has_hyper_embed:
    encoded_metadata['encoded_hyper'] = model.apply(
        {'params': params}, metadata, method=model.encode_hyper_embed)
  return encoded_metadata


#   def _loss_fn(params):
#     def _model_fn(coarse_key, fine_key, params, batch, extra_params): #state.extra_params
#       out = model.apply({'params': params},
#                         batch,
#                         extra_params=extra_params,
#                         return_points=(use_warp_reg_loss or use_hyper_reg_loss),
#                         return_weights=(use_warp_reg_loss or use_elastic_loss),
#                         return_warp_jacobian=use_elastic_loss,
#                         rngs={
#                             'coarse': coarse_key,
#                             'fine': fine_key
#                         },
#                         epsilon=epsilon)
#       return out
#     render_fn = functools.partial(render_image,
#                                   model_fn=_model_fn,
#                                   chunk=chunk_size)
#     ret = render_fn(state, batch, rng=rng_key)
#     losses = {}
#     stats = {}
#     losses['main'], stats['main'] = _compute_loss_and_stats(ret)

#     if use_background_loss:
#       background_loss = compute_background_loss(
#           model,
#           state=state,
#           params=params['model'],
#           key=reg_key,
#           points=batch['background_points'],
#           noise_std=scalar_params.background_noise_std)
#       background_loss = background_loss.mean()
#       losses['background'] = (
#           scalar_params.background_loss_weight * background_loss)
#       stats['background_loss'] = background_loss
#     return sum(losses.values()), (stats, ret)

#   optimizer = state.optimizer
  
#   grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#   (_, (stats, model_out)), grad = grad_fn(optimizer.target)
#   grad = jax.lax.pmean(grad, axis_name='batch')
#   stats = jax.lax.pmean(stats, axis_name='batch')
#   new_optimizer = optimizer.apply_gradient(
#       grad, learning_rate=scalar_params.learning_rate)
#   new_state = state.replace(optimizer=new_optimizer)
#   model_out = jax.lax.pmean(model_out, axis_name='batch')
#   return new_state, stats, rng_key, model_out, grad


# def encode_metadata(model, params, metadata):
#   """Encodes metadata embeddings.

#   Args:
#     model: a NerfModel.
#     params: the parameters of the model.
#     metadata: the metadata dict.

#   Returns:
#     A new metadata dict with the encoded embeddings.
#   """
#   encoded_metadata = {}
#   if model.use_nerf_embed:
#     encoded_metadata['encoded_nerf'] = model.apply(
#         {'params': params}, metadata, method=model.encode_nerf_embed)
#   if model.use_warp:
#     encoded_metadata['encoded_warp'] = model.apply(
#         {'params': params}, metadata, method=model.encode_warp_embed)
#   if model.has_hyper_embed:
#     encoded_metadata['encoded_hyper'] = model.apply(
#         {'params': params}, metadata, method=model.encode_hyper_embed)
#   return encoded_metadata

# #TODO THIS IS IT FOR RENDERING AN IMAGE
# def render_image(
#     state,
#     rays_dict,
#     model_fn,
#     rng,
#     chunk=8192):
#   """Render all the pixels of an image.

#   Args:
#     state: model_utils.TrainState.
#     rays_dict: dict, test example.
#     model_fn: function, jit-ed render function.
#     device_count: The number of devices to shard batches over.
#     rng: The random number generator.
#     chunk: int, the size of chunks to render sequentially.
#     default_ret_key: either 'fine' or 'coarse'. If None will default to highest.

#   Returns:
#     rgb: jnp.ndarray, rendered color image.
#     depth: jnp.ndarray, rendered depth.
#     acc: jnp.ndarray, rendered accumulated weights per pixel.
#   """
#   batch_shape = rays_dict['origins'].shape[:-1]
#   num_rays = np.prod(batch_shape)
#   rays_dict = tree_util.tree_map(lambda x: x.reshape((num_rays, -1)), rays_dict)
#   _, key_0, key_1 = jax.random.split(rng, 3)
#   ret_maps = []
#   num_batches = int(math.ceil(num_rays / chunk))
#   for batch_idx in range(num_batches):
#     ray_idx = batch_idx * chunk
#     # pylint: disable=cell-var-from-loop
#     chunk_slice_fn = lambda x: x[ray_idx:ray_idx + chunk]
#     chunk_rays_dict = tree_util.tree_map(chunk_slice_fn, rays_dict)
#     # After padding the number of chunk_rays is always divisible by
#     # proc_count.
#     #per_proc_rays = num_chunk_rays // jax.process_count()
#     #chunk_rays_dict = tree_util.tree_map(
#     #    lambda x: x[(proc_id * per_proc_rays):((proc_id + 1) * per_proc_rays)],
#     #    chunk_rays_dict)
#     model_out = model_fn(key_0, key_1, state.optimizer.target['model'],
#                          chunk_rays_dict, state.extra_params)
#     ret_maps.append(model_out)
#   ret_map = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), *ret_maps)
#   if 'fine' in ret_map.keys():
#     ret_map = ret_map['fine']
#   out = {}
#   for key, value in ret_map.items():
#     out_shape = (*batch_shape, *value.shape[1:])
#     out[key] = value.reshape(out_shape)

#   return out
