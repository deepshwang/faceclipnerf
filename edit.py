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

# Lint as: python3
"""Training script for Nerf."""

import functools
from typing import Dict, Union
import os


from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax import traverse_util
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf

from hypernerf import configs
from hypernerf import datasets
from hypernerf import gpath
from hypernerf import model_utils
from hypernerf import editing_models as models
from hypernerf import schedules
from hypernerf import editing as training
from hypernerf import utils
from hypernerf import image_utils
from hypernerf import clip

from datetime import datetime
import pytz

import debug 
import ipdb

from transformers import CLIPTokenizer, FlaxCLIPModel, CLIPProcessor


flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_string('ref_base_folder', None, 'where to call the original model from')
flags.mark_flag_as_required('ref_base_folder')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
flags.DEFINE_multi_string('target_text_prompt', None, 'Target text prompt for manipulation.')
flags.DEFINE_multi_string('lambda_alphatv', None, 'Strength for alpha TV regularization.')
flags.DEFINE_multi_string('reference_warp_id', None, 'Warp embed id to use.')
flags.DEFINE_multi_string('anchor_embedding_ids', None, 'Anchor embedding ids.')
FLAGS = flags.FLAGS


def _log_to_tensorboard(writer: tensorboard.SummaryWriter,
                        state: model_utils.TrainState,
                        scalar_params: training.ScalarParams,
                        stats: Dict[str, Union[Dict[str, jnp.ndarray],
                                               jnp.ndarray]],
                        time_dict: Dict[str, jnp.ndarray]):
  """Log statistics to Tensorboard."""
  step = int(state.optimizer.state.step)

  def _log_scalar(tag, value):
    if value is not None:
      writer.scalar(tag, value, step)

  _log_scalar('params/learning_rate', scalar_params.learning_rate)

  # pmean is applied in train_step so just take the item.
  for stat_key, stat_value in stats['main'].items():
    writer.scalar(f'{stat_key}', stat_value, step)

  _log_scalar('loss/background', stats.get('background_loss'))

  for k, v in time_dict.items():
    writer.scalar(f'time/{k}', v, step)


def _log_histograms(writer: tensorboard.SummaryWriter,
                    state: model_utils.TrainState,
                    model_out):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_embed' in params:
    embeddings = params['nerf_embed']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_embed' in params:
    embeddings = params['hyper_embed']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_embed' in params:
    embeddings = params['warp_embed']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)

  if 'warped_points' in model_out:
    points = model_out['points']
    warped_points = model_out['warped_points']
    writer.histogram('spatial_points',
                      warped_points[..., :3], step)
    writer.histogram('spatial_points_delta',
                      warped_points[..., :3] - points, step)
    if warped_points.shape[-1] > 3:
      writer.histogram('/hyper_points',
                        warped_points[..., 3:], step)


def _log_grads(writer: tensorboard.SummaryWriter, model: models.EditingNerfModel,
               state: model_utils.TrainState):
  """Log histograms to Tensorboard."""
  step = int(state.optimizer.state.step)
  params = state.optimizer.target['model']
  if 'nerf_metadata_encoder' in params:
    embeddings = params['nerf_metadata_encoder']['embed']['embedding']
    writer.histogram('nerf_embedding', embeddings, step)
  if 'hyper_metadata_encoder' in params:
    embeddings = params['hyper_metadata_encoder']['embed']['embedding']
    writer.histogram('hyper_embedding', embeddings, step)
  if 'warp_field' in params and model.warp_metadata_config['type'] == 'glo':
    embeddings = params['warp_metadata_encoder']['embed']['embedding']
    writer.histogram('warp_embedding', embeddings, step)


def main(argv):
  jax.config.parse_flags_with_absl()
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  # Assume G3 path for config files when running locally.
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))
  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  dummy_model = models.EditingNerfModel({}, 0, 0)
  
  # Replace configuration with FLAG values
  train_config.target_text_prompt = FLAGS.target_text_prompt[0]
  train_config.lambda_alphatv = float(FLAGS.lambda_alphatv[0])
  reference_warp_id = int(FLAGS.reference_warp_id[0])
  anchor_embedding_ids = [int(i) for i in FLAGS.anchor_embedding_ids[0].split(",")]
  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  #seoul_tz = pytz.timezone("Asia/Seoul")
  #now = datetime.now(seoul_tz)
  #dt_string = now.strftime("_%d-%m-%Y_%H:%M:%S")
  #exp_dir = gpath.GPath(str(exp_dir) + dt_string)
  summary_dir = exp_dir / 'summaries' / 'train'
  checkpoint_dir = exp_dir / 'checkpoints'
  # Retrieve reference model directory information
  proj_name = str(exp_dir).split("/")[-1].split("_")[1]
  #ref_checkpoint_dir = str(exp_dir).split("/")
  #ref_checkpoint_dir[-1] = proj_name
  #ref_checkpoint_dir = gpath.GPath("/".join(ref_checkpoint_dir)) / 'checkpoints'
  ref_checkpoint_dir = gpath.GPath(FLAGS.ref_base_folder) / 'checkpoints'
  # Save edited rgb for a random single train view for fast validation
  val_rgb_dir = exp_dir / 'fast_validation_images'
  os.makedirs(val_rgb_dir)
  # Log and create directories if this is the main process.
  if jax.process_index() == 0:
    logging.info('exp_dir = %s', exp_dir)
    if not exp_dir.exists():
      exp_dir.mkdir(parents=True, exist_ok=True)

    logging.info('summary_dir = %s', summary_dir)
    if not summary_dir.exists():
      summary_dir.mkdir(parents=True, exist_ok=True)

    logging.info('checkpoint_dir = %s', checkpoint_dir)
    if not checkpoint_dir.exists():
      checkpoint_dir.mkdir(parents=True, exist_ok=True)

  logging.info('Starting process %d. There are %d processes.',
               jax.process_index(), jax.process_count())
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))
  rng = random.PRNGKey(exp_config.random_seed)
  # Shift the numpy random seed by process_index() to shuffle data loaded by
  # different processes.
  np.random.seed(exp_config.random_seed + jax.process_index())

  #if train_config.batch_size % jax.device_count() != 0:
  #  raise ValueError('Batch size must be divisible by the number of devices.')
  devices = jax.local_devices()
  logging.info('Creating datasource')
  datasource = exp_config.datasource_cls(
      image_scale=exp_config.image_scale,
      random_seed=exp_config.random_seed,
      # Enable metadata based on model needs.
      use_warp_id=dummy_model.use_warp,
      use_appearance_id=(
          dummy_model.nerf_embed_key == 'appearance'
          or dummy_model.hyper_embed_key == 'appearance'),
      use_camera_id=dummy_model.nerf_embed_key == 'camera',
      use_time=dummy_model.warp_embed_key == 'time',
      reference_warp_id=reference_warp_id,
      reference_appearance_id=reference_warp_id,
      load_refrgb=True)
  # Create Jax iterator.
  logging.info('Creating dataset iterator.')
  train_iter = datasource.create_iterator(
      datasource.train_ids,
      #[format(9, '06d')],
      batch_size = -1 * len(jax.local_devices()),
      devices=devices,
      prefetch_size=2,
      square_crop=True
  )
  # Create Model.
  logging.info('Initializing models.')
  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
      key,
      batch_size=train_config.chunk_size,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)
  # Inject reference model parameter (pre-trained model) to our editing state object
  logging.info('Loading pretrained reference model params.')
  ref_state = checkpoints.restore_checkpoint(ref_checkpoint_dir, target=None)
  ref_params = ref_state['optimizer']['target']
  params = model_utils.inject_params(ref_params, params,
  reference_warp_id = reference_warp_id, 
  reference_hyper_id=[datasource.get_warp_id(format(i, '06d')) for i in anchor_embedding_ids])

  points_iter = None
  if train_config.use_background_loss:
    points = datasource.load_points(shuffle=True)
    points_batch_size = min(
        len(points),
        len(devices) * train_config.background_points_batch_size)
    points_batch_size -= points_batch_size % len(devices)
    points_dataset = tf.data.Dataset.from_tensor_slices(points)
    points_iter = datasets.iterator_from_dataset(
        points_dataset,
        batch_size=points_batch_size,
        prefetch_size=3,
        devices=devices)

  learning_rate_sched = schedules.from_config(train_config.lr_schedule)
  
  # Defining optimizer. Freeze all except HyperMapperMLP
  logging.info('Loading optimizers.')
  mapper_optimizer_def = optim.Adam(learning_rate_sched(0))
  mapper_traverser = traverse_util.ModelParamTraversal(lambda path, _: 'hyper_mapper_mlp' in path)
  optimizer_def = optim.MultiOptimizer((mapper_traverser, mapper_optimizer_def))


  optimizer = optimizer_def.create(params)
  state = model_utils.TrainState(
      optimizer=optimizer,
      nerf_alpha=None,
      warp_alpha=None,
      hyper_alpha=None,
      hyper_sheet_alpha=None)
  scalar_params = training.ScalarParams(
      learning_rate=learning_rate_sched(0),
      background_loss_weight=train_config.background_loss_weight,
      lambda_refrgb=train_config.lambda_refrgb,
      lambda_clip=train_config.lambda_clip,
      lambda_alphatv=train_config.lambda_alphatv)

  state = checkpoints.restore_checkpoint(checkpoint_dir, state)
  init_step = state.optimizer.state.step + 1
  state = jax_utils.replicate(state, devices=devices)
  del params

  text_features = clip.compute_delta_text_features(train_config.reference_text_prompt, train_config.target_text_prompt)
  #text_features = clip.compute_text_features(train_config.target_text_prompt)
  # Prepare summary writer 
  summary_writer = None
  if jax.process_index() == 0:
    config_str = gin.operative_config_str()
    logging.info('Configuration: \n%s', config_str)
    with (exp_dir / 'config.gin').open('w') as f:
      f.write(config_str)
    summary_writer = tensorboard.SummaryWriter(str(summary_dir))
    summary_writer.text('gin/train', textdata=gin.markdown(config_str), step=0)

  # Prepare trainer
  logging.info('Loading optimizers.')
  train_step = functools.partial(
      training.train_step,
      model,
      clip_text=text_features,
      use_elastic_loss=train_config.use_elastic_loss,
      use_background_loss=train_config.use_background_loss,
      use_warp_reg_loss=train_config.use_warp_reg_loss,
      use_hyper_reg_loss=train_config.use_hyper_reg_loss,
      chunk_size=train_config.chunk_size,
      epsilon=train_config.epsilon,
      use_mapper=train_config.use_mapper
  )

  ptrain_step = jax.pmap(
      train_step,
      axis_name='batch',
      devices=devices,
      # rng_key, state, batch, scalar_params.
      in_axes=(0, 0, 0, None),
      # Treat use_elastic_loss as compile-time static.
      donate_argnums=(1,),  # Donate the 'state' argument.
  )

  if devices:
    n_local_devices = len(devices)
  else:
    n_local_devices = jax.local_device_count()

  logging.info('Starting training')
  # Make random seed separate across processes.
  rng = rng + jax.process_index()
  keys = random.split(rng, n_local_devices)
  time_tracker = utils.TimeTracker()
  time_tracker.tic('data', 'total')
  for step, batch in zip(range(init_step, train_config.max_steps + 1),
                         train_iter):
    time_tracker.toc('data')
    #  See: b/162398046.
    # pytype: disable=attribute-error
    scalar_params = scalar_params.replace(
        learning_rate=learning_rate_sched(step))
    # pytype: enable=attribute-error
    with time_tracker.record_time('train_step'):
      state, stats, keys, model_out, grad = ptrain_step(
          keys, state, batch, scalar_params)
      time_tracker.toc('total')
    #image_utils.save_image("./testie.png", image_utils.image_to_uint8(jnp.reshape(model_out['fine']['rgb'][0], batch['rgb'].shape[1:])))
    if step % train_config.print_every == 0 and jax.process_index() == 0:
      logging.info('step=%d, %s', step,
                   time_tracker.summary_str('last'))
      fine_metrics_str = ', '.join(
          [f'{k}={v.mean():.04f}' for k, v in stats['main'].items()])
      logging.info('\tfine metrics: %s', fine_metrics_str)

    if step % train_config.save_every == 0 and jax.process_index() == 0:
      image_utils.save_image(val_rgb_dir / "testie_{}.png".format(str(step)), image_utils.image_to_uint8(jnp.reshape(model_out['fine']['rgb'][0, ..., :3], batch['rgb'].shape[1:]))) 
      for j in range(4):
        mask_img = model_out['fine']['mask'][0, ..., j]
        image_utils.save_image(val_rgb_dir / "testie_{}_mask_{}.png".format(str(step), str(j)), image_utils.image_to_uint8(jnp.reshape(mask_img, batch['rgb'].shape[1:3]))) 
      training.save_checkpoint(checkpoint_dir, state, keep=2)

    if step % train_config.log_every == 0 and jax.process_index() == 0:
      # Only log via process 0.
      _log_to_tensorboard(
          summary_writer,
          jax_utils.unreplicate(state),
          scalar_params,
          jax_utils.unreplicate(stats),
          time_dict=time_tracker.summary('mean'))
      time_tracker.reset()

    if step % train_config.histogram_every == 0 and jax.process_index() == 0:
      _log_histograms(summary_writer, jax_utils.unreplicate(state), model_out)

    time_tracker.tic('data', 'total')

  if train_config.max_steps % train_config.save_every != 0:
    training.save_checkpoint(checkpoint_dir, state, keep=2)


if __name__ == '__main__':
  app.run(main)

## hyper_sheet_mlp/mlp_0/logit