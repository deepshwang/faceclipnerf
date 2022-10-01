import glob
import ipdb
import cv2
import numpy as np
import os
from tqdm import tqdm

img_dir="/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5/rgb/8x/*.png"
save_dir="/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5/masked_rgb/8x"
mask_dir="/home/nas4_user/sungwonhwang/data/hypernerf/shwangv5/mask/8x/*.png"

#img_dir="/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1/rgb/8x/*.png"
#save_dir="/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1/masked_rgb/8x"
#mask_dir="/home/nas4_user/sungwonhwang/data/hypernerf/jhyungv1/mask/8x/*.png"


os.makedirs(save_dir, exist_ok=True)

imgs = sorted(glob.glob(img_dir))
masks = sorted(glob.glob(mask_dir))

for img, mask in tqdm(zip(imgs, masks), total=len(imgs)):
    fname = img.split("/")[-1]
    img = cv2.imread(img)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    mask = np.expand_dims((mask==0) + 0, axis=-1)
    out = img * mask
    save_name = save_dir + "/{}".format(fname)
    cv2.imwrite(save_name, out)