import cv2
import os
import numpy as np
import pdb
import imageio
from PIL import Image
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import configargparse

def save_image(path, image: np.ndarray) -> None:
  #print(f'Saving {path}')
  if not path.parent.exists():
    path.parent.mkdir(exist_ok=True, parents=True)
  with path.open('wb') as f:
    image = Image.fromarray(np.asarray(image))
    image.save(f, format=path.suffix.lstrip('.'))


def image_to_uint8(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint8 array."""
  if image.dtype == np.uint8:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
  """Trim the image if not divisible by the divisor."""
  height, width = image.shape[:2]
  if height % divisor == 0 and width % divisor == 0:
    return image

  new_height = height - height % divisor
  new_width = width - width % divisor

  return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Downsamples the image by an integer factor to prevent artifacts."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  if height % scale > 0 or width % scale > 0:
    raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                     f' scale ({scale}).')
  out_height, out_width = height // scale, width // scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized



parser = configargparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, 
                    help='data root')
parser.add_argument("--capture_name", type=str,
                    help='capture name')
args = parser.parse_args()


data_dir = os.path.expanduser(args.data_dir)
capture_name = args.capture_name
root_dir = Path(data_dir, capture_name)
videofile_name = args.capture_name + ".mov"
fps=60
target_num_frames=100
max_scale = 1.0 # adjust this to smaller value for faster processing

# PATHS
rgb_dir = root_dir /  "rgb"
rgb_raw_dir = root_dir /  "rgb-raw"

colmap_db_path = root_dir / "database.db"
colmap_out_path = root_dir /  "sparse"


def check_file_length():
    video_path = os.path.join(data_dir, capture_name) + "/" + videofile_name
    cap = cv2.VideoCapture(video_path)

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < target_num_frames:
        raise RuntimeError('The video is too short and has fewer frames than the target.')

    if fps == -1:
        fps = int(target_num_frames / num_frames * input_fps)
        print(f"Auto-computed FPS = {fps}")

if __name__ == "__main__":

    image_scales = [1, 2, 4, 8, 16]
    for image_path in tqdm(Path(rgb_raw_dir).glob('*.png')):
        image = make_divisible(imageio.imread(image_path), max(image_scales))
        for scale in image_scales:
            save_image(
        rgb_dir / f'{scale}x/{image_path.stem}.png',
        image_to_uint8(downsample_image(image, scale)))
