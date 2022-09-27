import torch
from human_segmentation.models import create_model, tf_albu
from tqdm import tqdm
import configargparse
from pathlib import Path
import os
import ipdb
import imageio
import torchvision.transforms as transforms
import torchvision

parser = configargparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, 
                    help='data root')
parser.add_argument("--capture_name", type=str,
                    help='capture name')
args = parser.parse_args()

model = create_model("Unet_2020-07-20").cuda()
tf = transforms.ToTensor()


data_dir = os.path.expanduser(args.data_dir)
capture_name = args.capture_name
root_dir = Path(data_dir, capture_name)

rgb_dir = root_dir /  "rgb"
rgb_raw_dir = root_dir /  "rgb-raw"

scales_dir = [rgb_dir / "{}x".format(i) for i in [1, 2, 4, 8, 16]]
for scale_dir in scales_dir:
    print("Extracting mask for scale {}".format(scale_dir))
    for image_path in tqdm(Path(scale_dir).glob("*.png")):
        img = imageio.imread(image_path)
        H, W, _ = img.shape

        #Resizer
        scale = 800 / max(H, W)
        H_scale = int(min(H * scale, 800))
        W_scale = int(min(W * scale, 800))
        resizer = transforms.Compose([transforms.CenterCrop((H_scale, W_scale)),
                                    transforms.Resize((H, W))])        
        img = torch.from_numpy(tf_albu(image=img)["image"]).unsqueeze(0).permute(0,3,1,2).cuda()
        mask = resizer(torch.squeeze(model(img), 0)) < 0.8
        mask = mask + 0. 
        mask_path = str(Path(str(image_path).replace("rgb", "mask"))) + ".png"
        os.makedirs("/".join(str(mask_path).split("/")[:-1]), exist_ok=True)
        torchvision.utils.save_image(mask, mask_path)
