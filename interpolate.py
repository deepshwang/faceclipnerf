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
"""Evaluation script for Nerf."""
import collections
import functools
import os
import time
from typing import Any, Dict, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
import gin
import jax
from jax import numpy as jnp
from jax import random
from jax.config import config
import numpy as np
import tensorflow as tf
import tensorflow_hub as tf_hub
import copy

from hypernerf import configs
from hypernerf import datasets
from hypernerf import evaluation
from hypernerf import gpath
from hypernerf import image_utils
from hypernerf import model_utils
from hypernerf import models
from hypernerf import types
from hypernerf import utils
from hypernerf import visualization as viz
from hypernerf import debug

from tqdm import tqdm
import cv2


import pdb


flags.DEFINE_enum('mode', None, ['jax_cpu', 'jax_gpu', 'jax_tpu'],
                  'Distributed strategy approach.')

flags.DEFINE_string('base_folder', None, 'where to store ckpts and logs')
flags.mark_flag_as_required('base_folder')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', (), 'Gin config files.')
flags.DEFINE_multi_string('start_warp_id', None, 'start warp id.')
flags.DEFINE_multi_string('end_warp_id', None, 'end warp id.')
FLAGS = flags.FLAGS

config.update('jax_log_compiles', True)


def compute_multiscale_ssim(image1: jnp.ndarray, image2: jnp.ndarray):
  """Compute the multiscale SSIM metric."""
  image1 = tf.convert_to_tensor(image1)
  image2 = tf.convert_to_tensor(image2)
  return tf.image.ssim_multiscale(image1, image2, max_val=1.0)


def compute_ssim(image1: jnp.ndarray, image2: jnp.ndarray, pad=0,
                 pad_mode='linear_ramp'):
  """Compute the LPIPS metric."""
  image1 = np.array(image1)
  image2 = np.array(image2)
  if pad > 0:
    image1 = image_utils.pad_image(image1, pad, pad_mode)
    image2 = image_utils.pad_image(image2, pad, pad_mode)
  psnr = tf.image.ssim(
      tf.convert_to_tensor(image1),
      tf.convert_to_tensor(image2),
      max_val=1.0)
  return np.asarray(psnr)


def compute_stats(batch, model_out):
  """Compute evaluation stats."""
  stats = {}
  rgb = model_out['rgb'][..., :3]

  if 'rgb' in batch:
    rgb_target = batch['rgb'][..., :3]
    mse = ((rgb - batch['rgb'][..., :3])**2).mean()
    psnr = utils.compute_psnr(mse)
    ssim = compute_ssim(rgb_target, rgb)
    ms_ssim = compute_multiscale_ssim(rgb_target, rgb)
    stats['mse'] = mse
    stats['psnr'] = psnr
    stats['ssim'] = ssim
    stats['ms_ssim'] = ms_ssim

    logging.info(
        '\tMetrics: mse=%.04f, psnr=%.02f, ssim=%.02f, ms_ssim=%.02f',
        mse, psnr, ssim, ms_ssim)

  stats = jax.tree_map(np.array, stats)

  return stats





def plot_images(*,
                batch: Dict[str, jnp.ndarray],
                tag: str,
                item_id: str,
                step: int,
                summary_writer: tensorboard.SummaryWriter,
                model_out: Any,
                save_dir: Optional[gpath.GPath],
                datasource: datasets.DataSource,
                extra_images=None):
  """Process and plot a single batch."""
  item_id = item_id.replace('/', '_')
  rgb = model_out['rgb'][..., :3]
  acc = model_out['acc']
  depth_exp = model_out['depth']
  depth_med = model_out['med_depth']
  colorize_depth = functools.partial(viz.colorize,
                                     cmin=datasource.near,
                                     cmax=datasource.far,
                                     invert=True)

  depth_exp_viz = colorize_depth(depth_exp)
  depth_med_viz = colorize_depth(depth_med)
  disp_exp_viz = viz.colorize(1.0 / depth_exp)
  disp_med_viz = viz.colorize(1.0 / depth_med)
  acc_viz = viz.colorize(acc, cmin=0.0, cmax=1.0)
  if save_dir:
    save_dir = save_dir / tag
    save_dir.mkdir(parents=True, exist_ok=True)
    image_utils.save_image(save_dir / f'rgb_{item_id}.png',
                           image_utils.image_to_uint8(rgb))
    image_utils.save_image(save_dir / f'depth_expected_viz_{item_id}.png',
                           image_utils.image_to_uint8(depth_exp_viz))
    image_utils.save_depth(save_dir / f'depth_expected_{item_id}.png',
                           depth_exp)
    image_utils.save_image(save_dir / f'depth_median_viz_{item_id}.png',
                           image_utils.image_to_uint8(depth_med_viz))
    image_utils.save_depth(save_dir / f'depth_median_{item_id}.png',
                           depth_med)

  summary_writer.image(f'rgb/{tag}/{item_id}', rgb, step)
  summary_writer.image(f'depth-expected/{tag}/{item_id}', depth_exp_viz, step)
  summary_writer.image(f'depth-median/{tag}/{item_id}', depth_med_viz, step)
  summary_writer.image(f'disparity-expected/{tag}/{item_id}', disp_exp_viz,
                       step)
  summary_writer.image(f'disparity-median/{tag}/{item_id}', disp_med_viz, step)
  summary_writer.image(f'acc/{tag}/{item_id}', acc_viz, step)

  if 'rgb' in batch:
    rgb_target = batch['rgb'][..., :3]
    rgb_abs_error = viz.colorize(
        abs(rgb_target - rgb).sum(axis=-1), cmin=0, cmax=1)
    rgb_sq_error = viz.colorize(
        ((rgb_target - rgb)**2).sum(axis=-1), cmin=0, cmax=1)
    summary_writer.image(f'rgb-target/{tag}/{item_id}', rgb_target, step)
    summary_writer.image(f'rgb-abs-error/{tag}/{item_id}', rgb_abs_error, step)
    summary_writer.image(f'rgb-sq-error/{tag}/{item_id}', rgb_sq_error, step)

  if 'depth' in batch:
    depth_target = batch['depth']
    depth_target_viz = colorize_depth(depth_target[..., 0])
    summary_writer.image(
        f'depth-target/{tag}/{item_id}', depth_target_viz, step)
    depth_med_error = viz.colorize(
        abs(depth_target - depth_med).squeeze(axis=-1), cmin=0, cmax=1)
    summary_writer.image(
        f'depth-median-error/{tag}/{item_id}', depth_med_error, step)
    depth_exp_error = viz.colorize(
        abs(depth_target - depth_exp).squeeze(axis=-1), cmin=0, cmax=1)
    summary_writer.image(
        f'depth-expected-error/{tag}/{item_id}', depth_exp_error, step)

  if extra_images:
    for k, v in extra_images.items():
      summary_writer.image(f'{k}/{tag}/{item_id}', v, step)


def sample_random_metadata(datasource, batch, step):
  """Samples random metadata dicts."""
  test_rng = random.PRNGKey(step)
  shape = batch['origins'][..., :1].shape
  metadata = {}
  if datasource.use_appearance_id:
    appearance_id = random.choice(
        test_rng, jnp.asarray(datasource.appearance_ids))
    logging.info('\tUsing appearance_id = %d', appearance_id)
    metadata['appearance'] = jnp.full(shape, fill_value=appearance_id,
                                      dtype=jnp.uint32)
  if datasource.use_warp_id:
    warp_id = random.choice(test_rng, jnp.asarray(datasource.warp_ids))
    logging.info('\tUsing warp_id = %d', warp_id)
    metadata['warp'] = jnp.full(shape, fill_value=warp_id, dtype=jnp.uint32)
  if datasource.use_camera_id:
    camera_id = random.choice(test_rng, jnp.asarray(datasource.camera_ids))
    logging.info('\tUsing camera_id = %d', camera_id)
    metadata['camera'] = jnp.full(shape, fill_value=camera_id,
                                  dtype=jnp.uint32)
  if datasource.use_time:
    timestamp = random.uniform(test_rng, minval=0.0, maxval=1.0)
    logging.info('\tUsing time = %d', timestamp)
    metadata['time'] = jnp.full(
        shape, fill_value=timestamp, dtype=jnp.uint32)
  return metadata


def process_iterator(tag: str,
                     item_ids: Sequence[str],
                     iterator,
                     rng: types.PRNGKey,
                     state: model_utils.TrainState,
                     step: int,
                     render_fn: Any,
                     summary_writer: tensorboard.SummaryWriter,
                     save_dir: Optional[gpath.GPath],
                     datasource: datasets.DataSource,
                     model: models.NerfModel,
                     start_warp_id: int,
                     end_warp_id: int):
  """Process a dataset iterator and compute metrics."""
  params = state.optimizer.target['model']
  interp_save_dir = save_dir / 'interp'  / '{}x'.format(datasource.image_scale) / '{}_{}'.format(start_warp_id, end_warp_id) 
  os.makedirs(interp_save_dir, exist_ok=True)
  meters = collections.defaultdict(utils.ValueMeter)
  for i, (item_id, batch) in tqdm(enumerate(zip(item_ids, iterator))):
    
    #if not os.path.exists(depth_filename) or not os.path.exists(refrgb_filename) or not os.path.exists(refrgb_imgname):
    if i==0:
      logging.info('[%s:%d/%d] Processing %s ', tag, i+1, len(item_ids), item_id)
      extra_images = None
      end_batch = copy.deepcopy(batch)
      end_batch['metadata']['warp']=  end_warp_id * np.ones_like(batch['metadata']['warp'])
      end_batch['metadata'] = evaluation.encode_metadata(model, jax_utils.unreplicate(params), end_batch['metadata'])
      batch['metadata'] = evaluation.encode_metadata(model, jax_utils.unreplicate(params), batch['metadata'])
      ratios = jnp.linspace(0.0, 1.0, num=61)
      video_dict = {'rgb':[]}
      for ratio in ratios:
        in_batch = copy.deepcopy(batch)
        noise_key = jax.random.PRNGKey(777)
        noise = jax.random.normal(noise_key, shape=(1, 1, 8))
        noise = noise / (jnp.linalg.norm(noise) * 10)
        #in_batch['metadata']['encoded_warp'] = (1.0 - ratio) * in_batch['metadata']['encoded_warp'] + ratio * end_batch['metadata']['encoded_warp']
        #in_batch['metadata']['encoded_hyper'] = (1.0 - ratio) * in_batch['metadata']['encoded_hyper'] + ratio * end_batch['metadata']['encoded_hyper'] + noise
        #in_batch['metadata']['encoded_warp'] = (1.0 - ratio) * in_batch['metadata']['encoded_warp'] + ratio * end_batch['metadata']['encoded_warp']
        in_batch['metadata']['encoded_hyper'] = (1.0 - ratio) * in_batch['metadata']['encoded_hyper'] + ratio * end_batch['metadata']['encoded_hyper']
        model_out = render_fn(state, in_batch, rng=rng)
        depth = model_out['depth']
        rgb = model_out['rgb'][..., :3]
        video_dict['rgb'].append(np.array(model_out['rgb']))
        disp_exp_viz = viz.colorize(1.0 / depth)
        interp_imgname = str(interp_save_dir /  "rgb_{:.2f}.png".format(ratio))
        interp_depth_imgname = str(interp_save_dir /  "depth_{}_{}.png".format(str(ratio), item_id))
        image_utils.save_image(interp_imgname,
                              image_utils.image_to_uint8(rgb))
        #image_utils.save_image(interp_depth_imgname,
        #                      image_utils.image_to_uint8(disp_exp_viz))
      
      size = video_dict['rgb'][0].shape[:2]
      for k, frame_list in video_dict.items():
        fname = str(interp_save_dir / "{}.mp4".format(k))
        print("Writing test camera video at: ", fname)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(fname, fourcc, 20.0, (size[1], size[0]))
        for frame in frame_list:
          frame = (255*frame).astype(np.uint8)
          frame = frame[...,::-1]
          out.write(frame)
        out.release()    

def delete_old_renders(render_dir, max_renders):
  render_paths = sorted(render_dir.iterdir())
  paths_to_delete = render_paths[:-max_renders]
  for path in paths_to_delete:
    logging.info('Removing render directory %s', str(path))
    path.rmtree()


def main(argv):
  jax.config.parse_flags_with_absl()
  tf.config.experimental.set_visible_devices([], 'GPU')
  del argv
  logging.info('*** Starting experiment')
  gin_configs = FLAGS.gin_configs

  logging.info('*** Loading Gin configs from: %s', str(gin_configs))
  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True)

  # Load configurations.
  exp_config = configs.ExperimentConfig()
  train_config = configs.TrainConfig()
  eval_config = configs.EvalConfig()
  start_warp_id = int(FLAGS.start_warp_id[0])
  end_warp_id = int(FLAGS.end_warp_id[0])

  # Get directory information.
  exp_dir = gpath.GPath(FLAGS.base_folder)
  if exp_config.subname:
    exp_dir = exp_dir / exp_config.subname
  logging.info('\texp_dir = %s', exp_dir)
  if not exp_dir.exists():
    exp_dir.mkdir(parents=True, exist_ok=True)

  if eval_config.subname:
    summary_dir = exp_dir / 'summaries' / f'eval-{eval_config.subname}'
  else:
    summary_dir = exp_dir / 'summaries' / 'eval'
  logging.info('\tsummary_dir = %s', summary_dir)
  if not summary_dir.exists():
    summary_dir.mkdir(parents=True, exist_ok=True)

  if eval_config.subname:
    renders_dir = exp_dir / f'renders-{eval_config.subname}'
  else:
    renders_dir = exp_dir / 'renders'
  logging.info('\trenders_dir = %s', renders_dir)
  if not renders_dir.exists():
    renders_dir.mkdir(parents=True, exist_ok=True)

  checkpoint_dir = exp_dir / 'checkpoints'
  logging.info('\tcheckpoint_dir = %s', checkpoint_dir)

  logging.info('Starting process %d. There are %d processes.',
               jax.process_index(), jax.process_count())
  logging.info('Found %d accelerator devices: %s.', jax.local_device_count(),
               str(jax.local_devices()))
  logging.info('Found %d total devices: %s.', jax.device_count(),
               str(jax.devices()))

  rng = random.PRNGKey(20200823)

  devices_to_use = jax.local_devices()
  n_devices = len(
      devices_to_use) if devices_to_use else jax.local_device_count()

  logging.info('Creating datasource')
  # Dummy model for configuratin datasource.
  dummy_model = models.NerfModel({}, 0, 0)
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
      reference_warp_id=start_warp_id,
      reference_appearance_id=start_warp_id)

  # Get training IDs to evaluate.
  train_eval_iter = datasource.create_iterator(datasource.train_ids, batch_size=0)

  rng, key = random.split(rng)
  params = {}
  model, params['model'] = models.construct_nerf(
      key,
      batch_size=eval_config.chunk,
      embeddings_dict=datasource.embeddings_dict,
      near=datasource.near,
      far=datasource.far)

  optimizer_def = optim.Adam(0.0)
  if train_config.use_weight_norm:
    optimizer_def = optim.WeightNorm(optimizer_def)
  optimizer = optimizer_def.create(params)
  init_state = model_utils.TrainState(optimizer=optimizer)
  del params

  def _model_fn(key_0, key_1, params, rays_dict, extra_params):
    out = model.apply({'params': params},
                      rays_dict,
                      extra_params=extra_params,
                      metadata_encoded=True,
                      rngs={
                          'coarse': key_0,
                          'fine': key_1
                      },
                      mutable=False)
    return jax.lax.all_gather(out, axis_name='batch')

  pmodel_fn = jax.pmap(
      # Note rng_keys are useless in eval mode since there's no randomness.
      _model_fn,
      in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
      devices=devices_to_use,
      axis_name='batch',
  )

  render_fn = functools.partial(evaluation.render_image,
                                model_fn=pmodel_fn,
                                device_count=n_devices,
                                chunk=eval_config.chunk)

  last_step = 0
  summary_writer = tensorboard.SummaryWriter(str(summary_dir))

  state = checkpoints.restore_checkpoint(checkpoint_dir, init_state)
  state = jax_utils.replicate(state, devices=devices_to_use)
  step = int(state.optimizer.state.step[0])

  save_dir = renders_dir if eval_config.save_output else None
  process_iterator(tag='train',
                    item_ids=datasource.train_ids,
                    iterator=train_eval_iter,
                    state=state,
                    rng=rng,
                    step=step,
                    render_fn=render_fn,
                    summary_writer=summary_writer,
                    save_dir=save_dir,
                    datasource=datasource,
                    model=model,
                    start_warp_id=start_warp_id,
                    end_warp_id=end_warp_id)

  if save_dir:
    delete_old_renders(renders_dir, eval_config.max_render_checkpoints)



if __name__ == '__main__':
  app.run(main)
