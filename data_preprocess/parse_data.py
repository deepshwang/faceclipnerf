# @title Define Scene Manager.
from absl import logging
from typing import Dict
import numpy as np
from nerfies.camera import Camera
import pycolmap
from pycolmap import Quaternion
import imageio
import plotly.graph_objs as go
from pathlib import Path
import configargparse
import jax
from jax import numpy as jnp 
from tensorflow_graphics.geometry.representation.ray import triangulate as ray_triangulate
import cv2
import mediapipe as mp
from PIL import Image
from scipy import linalg
import pandas as pd
import math
from scipy import interpolate
from pprint import pprint
import json
import ipdb


use_face = True  # @param {type: 'boolean'}

def convert_colmap_camera(colmap_camera, colmap_image):
  """Converts a pycolmap `image` to an SFM camera."""
  camera_rotation = colmap_image.R()
  camera_position = -(colmap_image.t @ camera_rotation)
  new_camera = Camera(
      orientation=camera_rotation,
      position=camera_position,
      focal_length=colmap_camera.fx,
      pixel_aspect_ratio=colmap_camera.fx / colmap_camera.fx,
      principal_point=np.array([colmap_camera.cx, colmap_camera.cy]),
      radial_distortion=np.array([colmap_camera.k1, colmap_camera.k2, 0.0]),
      tangential_distortion=np.array([colmap_camera.p1, colmap_camera.p2]),
      skew=0.0,
      image_size=np.array([colmap_camera.width, colmap_camera.height])
  )
  return new_camera


def filter_outlier_points(points, inner_percentile):
  """Filters outlier points."""
  outer = 1.0 - inner_percentile
  lower = outer / 2.0
  upper = 1.0 - lower
  centers_min = np.quantile(points, lower, axis=0)
  centers_max = np.quantile(points, upper, axis=0)
  result = points.copy()

  too_near = np.any(result < centers_min[None, :], axis=1)
  too_far = np.any(result > centers_max[None, :], axis=1)

  return result[~(too_near | too_far)]


def average_reprojection_errors(points, pixels, cameras):
  """Computes the average reprojection errors of the points."""
  cam_errors = []
  for i, camera in enumerate(cameras):
    cam_error = reprojection_error(points, pixels[:, i], camera)
    cam_errors.append(cam_error)
  cam_error = np.stack(cam_errors)

  return cam_error.mean(axis=1)


def _get_camera_translation(camera):
  """Computes the extrinsic translation of the camera."""
  rot_mat = camera.orientation
  return -camera.position.dot(rot_mat.T)


def _transform_camera(camera, transform_mat):
  """Transforms the camera using the given transformation matrix."""
  # The determinant gives us volumetric scaling factor.
  # Take the cube root to get the linear scaling factor.
  scale = np.cbrt(np.linalg.det(transform_mat[:, :3]))
  quat_transform = ~Quaternion.FromR(transform_mat[:, :3] / scale)

  translation = _get_camera_translation(camera)
  rot_quat = Quaternion.FromR(camera.orientation)
  rot_quat *= quat_transform
  translation = scale * translation - rot_quat.ToR().dot(transform_mat[:, 3])
  new_transform = np.eye(4)
  new_transform[:3, :3] = rot_quat.ToR()
  new_transform[:3, 3] = translation

  rotation = rot_quat.ToR()
  new_camera = camera.copy()
  new_camera.orientation = rotation
  new_camera.position = -(translation @ rotation)
  return new_camera


def _pycolmap_to_sfm_cameras(manager: pycolmap.SceneManager) -> Dict[int, Camera]:
  """Creates SFM cameras."""
  # Use the original filenames as indices.
  # This mapping necessary since COLMAP uses arbitrary numbers for the
  # image_id.
  image_id_to_colmap_id = {
      image.name.split('.')[0]: image_id
      for image_id, image in manager.images.items()
  }

  sfm_cameras = {}
  for image_id in image_id_to_colmap_id:
    colmap_id = image_id_to_colmap_id[image_id]
    image = manager.images[colmap_id]
    camera = manager.cameras[image.camera_id]
    sfm_cameras[image_id] = convert_colmap_camera(camera, image)

  return sfm_cameras


class SceneManager:
  """A thin wrapper around pycolmap."""

  @classmethod
  def from_pycolmap(cls, colmap_path, image_path, min_track_length=10):
    """Create a scene manager using pycolmap."""
    manager = pycolmap.SceneManager(str(colmap_path))
    manager.load_cameras()
    manager.load_images()
    manager.load_points3D()
    manager.filter_points3D(min_track_len=min_track_length)
    sfm_cameras = _pycolmap_to_sfm_cameras(manager)
    points, point_colors = manager.get_filtered_points3D(return_colors=True)
    return cls(sfm_cameras, points, image_path, point_colors=point_colors)

  def __init__(self, cameras, points, image_path, point_colors=None):
    self.image_path = Path(image_path)
    self.camera_dict = cameras
    self.points = points
    self.point_colors = point_colors

    logging.info('Created scene manager with %d cameras', len(self.camera_dict))

  def __len__(self):
    return len(self.camera_dict)

  @property
  def image_ids(self):
    return sorted(self.camera_dict.keys())

  @property
  def camera_list(self):
    return [self.camera_dict[i] for i in self.image_ids]

  @property
  def camera_positions(self):
    """Returns an array of camera positions."""
    return np.stack([camera.position for camera in self.camera_list])

  def load_image(self, image_id):
    """Loads the image with the specified image_id."""
    path = self.image_path / f'{image_id}.png'
    with path.open('rb') as f:
      return imageio.imread(f)

  def triangulate_pixels(self, pixels):
    """Triangulates the pixels across all cameras in the scene.

    Args:
      pixels: the pixels to triangulate. There must be the same number of pixels
        as cameras in the scene.

    Returns:
      The 3D points triangulated from the pixels.
    """
    if pixels.shape != (len(self), 2):
      raise ValueError(
          f'The number of pixels ({len(pixels)}) must be equal to the number '
          f'of cameras ({len(self)}).')

    return triangulate_pixels(pixels, self.camera_list)

  def change_basis(self, axes, center):
    """Change the basis of the scene.

    Args:
      axes: the axes of the new coordinate frame.
      center: the center of the new coordinate frame.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    transform_mat = np.zeros((3, 4))
    transform_mat[:3, :3] = axes.T
    transform_mat[:, 3] = -(center @ axes)
    return self.transform(transform_mat)

  def transform(self, transform_mat):
    """Transform the scene using a transformation matrix.

    Args:
      transform_mat: a 3x4 transformation matrix representation a
        transformation.

    Returns:
      A new SceneManager with transformed points and cameras.
    """
    if transform_mat.shape != (3, 4):
      raise ValueError('transform_mat should be a 3x4 transformation matrix.')

    points = None
    if self.points is not None:
      points = self.points.copy()
      points = points @ transform_mat[:, :3].T + transform_mat[:, 3]

    new_cameras = {}
    for image_id, camera in self.camera_dict.items():
      new_cameras[image_id] = _transform_camera(camera, transform_mat)

    return SceneManager(new_cameras, points, self.image_path)

  def filter_images(self, image_ids):
    num_filtered = 0
    for image_id in image_ids:
      if self.camera_dict.pop(image_id, None) is not None:
        num_filtered += 1

    return num_filtered

# @title Load COLMAP scene.
if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        help='data dir')
    parser.add_argument("--colmap_image_scale", type=int,
                        default=1, help='rgb_dir')                        
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    colmap_dir = data_dir / 'sparse/0'
    rgb_dir = data_dir / 'rgb' 
    colmap_image_scale = args.colmap_image_scale

    scene_manager = SceneManager.from_pycolmap(
        colmap_dir, 
        rgb_dir / f'1x', 
        min_track_length=5)

    if colmap_image_scale > 1:
        print(f'Scaling COLMAP cameras back to 1x from {colmap_image_scale}x.')
    for item_id in scene_manager.image_ids:
        camera = scene_manager.camera_dict[item_id]
        scene_manager.camera_dict[item_id] = camera.scale(colmap_image_scale)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=scene_manager.points[:, 0],
        y=scene_manager.points[:, 1],
        z=scene_manager.points[:, 2],
        mode='markers',
        marker=dict(size=2,
        color=list(map(lambda e: 'rgb('+', '.join(e.astype(str))+')', scene_manager.point_colors)),
    )))
    fig.add_trace(go.Scatter3d(
        x=scene_manager.camera_positions[:, 0],
        y=scene_manager.camera_positions[:, 1],
        z=scene_manager.camera_positions[:, 2],
        mode='markers',
        marker=dict(size=2,
    )))
    fig.update_layout(scene_dragmode='orbit')
    fig.write_image(data_dir / "COLMAP_scene.png")
    np.save(data_dir / 'points.npy', scene_manager.points)
    def variance_of_laplacian(image: np.ndarray) -> np.ndarray:
      """Compute the variance of the Laplacian which measure the focus."""
      gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
      return cv2.Laplacian(gray, cv2.CV_64F).var()


    blur_filter_perc = 95.0 # @param {type: 'number'}
    if blur_filter_perc > 0.0:
      image_paths = sorted(rgb_dir.iterdir())
      print('Loading images.')
      images = list(map(scene_manager.load_image, scene_manager.image_ids))
      print('Computing blur scores.')
      blur_scores = np.array([variance_of_laplacian(im) for im in images])
      blur_thres = np.percentile(blur_scores, blur_filter_perc)
      blur_filter_inds = np.where(blur_scores >= blur_thres)[0]
      blur_filter_scores = [blur_scores[i] for i in blur_filter_inds]
      blur_filter_inds = blur_filter_inds[np.argsort(blur_filter_scores)]
      blur_filter_scores = np.sort(blur_filter_scores)
      blur_filter_image_ids = [scene_manager.image_ids[i] for i in blur_filter_inds]
      print(f'Filtering {len(blur_filter_image_ids)} IDs: {blur_filter_image_ids}')
      num_filtered = scene_manager.filter_images(blur_filter_image_ids)
      print(f'Filtered {num_filtered} images')

    # Face landmark
    if use_face:
      mp_face_mesh = mp.solutions.face_mesh
      mp_drawing = mp.solutions.drawing_utils 
      drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
      
      # Initialize MediaPipe Face Mesh.
      face_mesh = mp_face_mesh.FaceMesh(
          static_image_mode=True,
          max_num_faces=2,
          min_detection_confidence=0.5)
      
      
      def compute_landmarks(image):
        height, width = image.shape[:2]
        results = face_mesh.process(image)
        if results.multi_face_landmarks is None:
          return None
        # Choose first face found.
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks = np.array(
            [(o.x * width, o.y * height) for o in landmarks],
            dtype=np.uint32)
        return landmarks

      landmarks_dict = {}
      for item_id in scene_manager.image_ids:
        image = scene_manager.load_image(item_id)
        landmarks = compute_landmarks(image)
        if landmarks is not None:
          landmarks_dict[item_id] = landmarks
      
      landmark_item_ids = sorted(landmarks_dict)
      landmarks_pixels = np.array([landmarks_dict[i] for i in landmark_item_ids])
      landmarks_cameras = [scene_manager.camera_dict[i] for i in landmark_item_ids]
    
     
    #Triangulage landmarks in 3D
    if use_face:
      def compute_camera_rays(points, camera):
        origins = np.broadcast_to(camera.position[None, :], (points.shape[0], 3))
        directions = camera.pixels_to_rays(points.astype(jnp.float32))
        endpoints = origins + directions
        return origins, endpoints
      
      
      def triangulate_landmarks(landmarks, cameras):
        all_origins = []
        all_endpoints = []
        nan_inds = []
        for i, (camera_landmarks, camera) in enumerate(zip(landmarks, cameras)):
          origins, endpoints = compute_camera_rays(camera_landmarks, camera)
          if np.isnan(origins).sum() > 0.0 or np.isnan(endpoints).sum() > 0.0:
            continue
          all_origins.append(origins)
          all_endpoints.append(endpoints)
        all_origins = np.stack(all_origins, axis=-2).astype(np.float32)
        all_endpoints = np.stack(all_endpoints, axis=-2).astype(np.float32)
        weights = np.ones(all_origins.shape[:2], dtype=np.float32)
        points = np.array(ray_triangulate(all_origins, all_endpoints, weights))
      
        return points
      

      landmark_points = triangulate_landmarks(landmarks_pixels, landmarks_cameras)
    else:
      landmark_points = None

    DEFAULT_IPD = 0.06
    NOSE_TIP_IDX = 1
    FOREHEAD_IDX = 10
    CHIN_IDX = 152
    RIGHT_EYE_IDX = 145
    LEFT_EYE_IDX = 385
    RIGHT_TEMPLE_IDX = 162
    LEFT_TEMPLE_IDX = 389


    def _normalize(x):
      return x / linalg.norm(x)


    def fit_plane_normal(points):
      """Fit a plane to the points and return the normal."""
      centroid = points.sum(axis=0) / points.shape[0]
      _, _, vh = linalg.svd(points - centroid)
      return vh[2, :]


    def metric_scale_from_ipd(landmark_points, reference_ipd):
      """Infer the scene-to-metric conversion ratio from facial landmarks."""
      left_eye = landmark_points[LEFT_EYE_IDX]
      right_eye = landmark_points[RIGHT_EYE_IDX]
      model_ipd = linalg.norm(left_eye - right_eye)
      return reference_ipd / model_ipd


    def basis_from_landmarks(landmark_points):
      """Computes an orthonormal basis from facial landmarks."""
      # Estimate Z by fitting a plane
      # This works better than trusting the chin to forehead vector, especially in
      # full body captures.
      face_axis_z = _normalize(fit_plane_normal(landmark_points))
      face_axis_y = _normalize(landmark_points[FOREHEAD_IDX] -
                              landmark_points[CHIN_IDX])
      face_axis_x = _normalize(landmark_points[LEFT_TEMPLE_IDX] -
                              landmark_points[RIGHT_TEMPLE_IDX])

      # Fitted plane normal might be flipped. Check using a heuristic and flip it if
      # it's flipped.
      z_flipped = np.dot(np.cross(face_axis_x, face_axis_y), face_axis_z)
      if z_flipped < 0.0:
        face_axis_z *= -1

      # Ensure axes are orthogonal, with the Z axis being fixed.
      face_axis_y = np.cross(face_axis_z, face_axis_x)
      face_axis_x = np.cross(face_axis_y, face_axis_z)

      return np.stack([face_axis_x, face_axis_y, face_axis_z]).T


    if use_face:
      face_basis = basis_from_landmarks(landmark_points)
      new_scene_manager = scene_manager.change_basis(
          face_basis, landmark_points[NOSE_TIP_IDX])
      new_cameras = [new_scene_manager.camera_dict[i] for i in landmark_item_ids]
      new_landmark_points = triangulate_landmarks(landmarks_pixels, new_cameras)
      face_basis = basis_from_landmarks(landmark_points)
      scene_to_metric = metric_scale_from_ipd(landmark_points, DEFAULT_IPD)
      
      print(f'Computed basis: {face_basis}')
      print(f'Estimated metric scale = {scene_to_metric:.02f}')
    else:
      new_scene_manager = scene_manager


    def estimate_near_far_for_image(scene_manager, image_id):
      """Estimate near/far plane for a single image based via point cloud."""
      points = filter_outlier_points(scene_manager.points, 0.95)
      points = np.concatenate([
          points,
          scene_manager.camera_positions,
      ], axis=0)
      camera = scene_manager.camera_dict[image_id]
      pixels = camera.project(points)
      depths = camera.points_to_local_points(points)[..., 2]

      # in_frustum = camera.ArePixelsInFrustum(pixels)
      in_frustum = (
          (pixels[..., 0] >= 0.0)
          & (pixels[..., 0] <= camera.image_size_x)
          & (pixels[..., 1] >= 0.0)
          & (pixels[..., 1] <= camera.image_size_y))
      depths = depths[in_frustum]

      in_front_of_camera = depths > 0
      depths = depths[in_front_of_camera]

      near = np.quantile(depths, 0.001)
      far = np.quantile(depths, 0.999)

      return near, far


    def estimate_near_far(scene_manager):
      """Estimate near/far plane for a set of randomly-chosen images."""
      # image_ids = sorted(scene_manager.images.keys())
      image_ids = scene_manager.image_ids
      rng = np.random.RandomState(0)
      image_ids = rng.choice(
          image_ids, size=len(scene_manager.camera_list), replace=False)
      
      result = []
      for image_id in image_ids:
        near, far = estimate_near_far_for_image(scene_manager, image_id)
        result.append({'image_id': image_id, 'near': near, 'far': far})
      result = pd.DataFrame.from_records(result)
      return result


    near_far = estimate_near_far(new_scene_manager)
    print('Statistics for near/far computation:')
    print(near_far.describe())
    print()

    near = (near_far['near'].quantile(0.001) / 0.8)/10
    far = near_far['far'].quantile(0.999) * 1.2
    print('Selected near/far values:')
    print(f'Near = {near:.04f}')
    print(f'far = {far:.04f}')


    def get_bbox_corners(points):
      lower = points.min(axis=0)
      upper = points.max(axis=0)
      return np.stack([lower, upper])


    points = filter_outlier_points(new_scene_manager.points, 0.95)
    bbox_corners = get_bbox_corners(
        np.concatenate([points, new_scene_manager.camera_positions], axis=0))

    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print(f'Scene Center: {scene_center}')
    print(f'Scene Scale: {scene_scale}')

    ## Generate test cameras
    _EPSILON = 1e-5


    def points_bound(points):
      """Computes the min and max dims of the points."""
      min_dim = np.min(points, axis=0)
      max_dim = np.max(points, axis=0)
      return np.stack((min_dim, max_dim), axis=1)


    def points_centroid(points):
      """Computes the centroid of the points from the bounding box."""
      return points_bound(points).mean(axis=1)


    def points_bounding_size(points):
      """Computes the bounding size of the points from the bounding box."""
      bounds = points_bound(points)
      return np.linalg.norm(bounds[:, 1] - bounds[:, 0])


    def look_at(camera,
                camera_position: np.ndarray,
                look_at_position: np.ndarray,
                up_vector: np.ndarray):
      look_at_camera = camera.copy()
      optical_axis = look_at_position - camera_position
      norm = np.linalg.norm(optical_axis)
      if norm < _EPSILON:
        raise ValueError('The camera center and look at position are too close.')
      optical_axis /= norm

      right_vector = np.cross(optical_axis, up_vector)
      norm = np.linalg.norm(right_vector)
      if norm < _EPSILON:
        raise ValueError('The up-vector is parallel to the optical axis.')
      right_vector /= norm

      # The three directions here are orthogonal to each other and form a right
      # handed coordinate system.
      camera_rotation = np.identity(3)
      camera_rotation[0, :] = right_vector
      camera_rotation[1, :] = np.cross(optical_axis, right_vector)
      camera_rotation[2, :] = optical_axis

      look_at_camera.position = camera_position
      look_at_camera.orientation = camera_rotation
      return look_at_camera


    def compute_camera_rays(points, camera):
      origins = np.broadcast_to(camera.position[None, :], (points.shape[0], 3))
      directions = camera.pixels_to_rays(points.astype(jnp.float32))
      endpoints = origins + directions
      return origins, endpoints


    def triangulate_rays(origins, directions):
      origins = origins[np.newaxis, ...].astype('float32')
      directions = directions[np.newaxis, ...].astype('float32')
      weights = np.ones(origins.shape[:2], dtype=np.float32)
      points = np.array(ray_triangulate(origins, origins + directions, weights))
      return points.squeeze()


    ref_cameras = [c for c in new_scene_manager.camera_list]
    origins = np.array([c.position for c in ref_cameras])
    directions = np.array([c.optical_axis for c in ref_cameras])
    look_at = triangulate_rays(origins, directions)
    print('look_at', look_at)

    avg_position = np.mean(origins, axis=0)
    print('avg_position', avg_position)

    up = -np.mean([c.orientation[..., 1] for c in ref_cameras], axis=0)
    print('up', up)

    bounding_size = points_bounding_size(origins) / 2
    x_scale =   0.75# @param {type: 'number'}
    y_scale = 0.75  # @param {type: 'number'}
    xs = x_scale * bounding_size
    ys = y_scale * bounding_size
    radius = 0.75  # @param {type: 'number'}
    num_frames = 100  # @param {type: 'number'}


    origin = np.zeros(3)

    ref_camera = ref_cameras[0]
    print(ref_camera.position)
    z_offset = -0.1

    angles = np.linspace(0, 2*math.pi, num=num_frames)
    positions = []
   
    # Circular Movement
    y_ratio = 0.6
    for angle in angles:
      x = np.cos(angle) * radius * xs
      y = np.sin(angle) * radius * ys
      # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
      # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

      position = np.array([x, y_ratio * y, z_offset])
      # Make distance to reference point constant.
      position = avg_position + position
      positions.append(position)

    # Horizontal Movements
    for angle in angles:
      x = np.cos(angle) * radius * xs
      y = 0.0
      # x = xs * radius * np.cos(angle) / (1 + np.sin(angle) ** 2)
      # y = ys * radius * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle) ** 2)

      position = np.array([x, y, z_offset])
      # Make distance to reference point constant.
      position = avg_position + position
      positions.append(position)

    positions = np.stack(positions)

    orbit_cameras = []
    for position in positions:
      camera = ref_camera.look_at(position, look_at, up)
      orbit_cameras.append(camera)

    camera_paths = {'orbit-mild': orbit_cameras}


    scene_json_path = data_dir / 'scene.json'
    with scene_json_path.open('w') as f:
      json.dump({
          'scale': scene_scale,
          'center': scene_center.tolist(),
          'bbox': bbox_corners.tolist(),
          'near': near * scene_scale,
          'far': far * scene_scale,
      }, f, indent=2)

    print(f'Saved scene information to {scene_json_path}')


    all_ids = scene_manager.image_ids
    val_ids = all_ids[::20]
    train_ids = sorted(set(all_ids) - set(val_ids))
    dataset_json = {
        'count': len(scene_manager),
        'num_exemplars': len(train_ids),
        'ids': scene_manager.image_ids,
        'train_ids': train_ids,
        'val_ids': val_ids,
    }

    dataset_json_path = data_dir / 'dataset.json'
    with dataset_json_path.open('w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f'Saved dataset information to {dataset_json_path}')


    import bisect

    metadata_json = {}
    for i, image_id in enumerate(train_ids):
      metadata_json[image_id] = {
          'warp_id': i,
          'appearance_id': i,
          'camera_id': 0,
      }
    for i, image_id in enumerate(val_ids):
      i = bisect.bisect_left(train_ids, image_id)
      metadata_json[image_id] = {
          'warp_id': i,
          'appearance_id': i,
          'camera_id': 0,
      }

    metadata_json_path = data_dir / 'metadata.json'
    with metadata_json_path.open('w') as f:
        json.dump(metadata_json, f, indent=2)

    print(f'Saved metadata information to {metadata_json_path}')

    camera_dir = data_dir / 'camera'
    camera_dir.mkdir(exist_ok=True, parents=True)
    for item_id, camera in new_scene_manager.camera_dict.items():
      camera_path = camera_dir / f'{item_id}.json'
      print(f'Saving camera to {camera_path!s}')
      with camera_path.open('w') as f:
        json.dump(camera.to_json(), f, indent=2)

    test_camera_dir = data_dir / 'camera-paths'
    for test_path_name, test_cameras in camera_paths.items():
      out_dir = test_camera_dir / test_path_name
      out_dir.mkdir(exist_ok=True, parents=True)
      for i, camera in enumerate(test_cameras):
        camera_path = out_dir / f'{i:06d}.json'
        print(f'Saving camera to {camera_path!s}')
        with camera_path.open('w') as f:
          json.dump(camera.to_json(), f, indent=2)