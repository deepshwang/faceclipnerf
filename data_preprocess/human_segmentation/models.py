import re
from typing import Any, Dict, Optional, Union
from collections import namedtuple
from torch import nn
from torch.utils import model_zoo
from segmentation_models_pytorch import Unet
import yaml
from albumentations.core.serialization import from_dict

def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

model = namedtuple("model", ["url", "model"])

with open("human_segmentation/config.yaml") as f:
    hparams = yaml.load(f, Loader=yaml.SafeLoader)

tf_albu = from_dict(hparams["val_aug"])

models = {
    "Unet_2020-07-20": model(
        url="https://github.com/ternaus/people_segmentation/releases/download/0.0.1/2020-09-23a.zip",
        model=Unet(encoder_name="timm-efficientnet-b3", classes=1, encoder_weights=None),
    )
}


def create_model(model_name: str) -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)
    return model
