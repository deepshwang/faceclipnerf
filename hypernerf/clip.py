import numpy as np
import jax.numpy as jnp
from transformers import CLIPTokenizer, FlaxCLIPModel, CLIPProcessor
import ipdb
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

clip_model = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
 

def extract_augmented_text_features(text_prompt):
    '''
    Args:
        text_prompt: str | Source text prompt to be augmented.
    
    Return:
        features: list   | A list of numpy features of augmented text prompts.
    '''
    augmented_inputs = [temp.format(text_prompt) for temp in imagenet_templates]
    augmented_tokens = tokenizer(augmented_inputs, padding=True, return_tensors="np")
    features = clip_model.get_text_features(**augmented_tokens)
    features = features / jnp.linalg.norm(features, axis=1)[..., jnp.newaxis]
    return features

def compute_delta_text_features(r_prompt, t_prompt):
    '''
    Args:
        r_prompt: str                  | Reference prompt (neutral).
        t_prompt: str                  | Target prompt for manipulation. 
    
    Return:
        delta_feature: (512,) np.array | Difference of averaged text embeddings in CLIP space.
    '''

    avg_r_feature = jnp.mean(jnp.stack(extract_augmented_text_features(r_prompt), axis=0), axis=0)
    avg_t_feature = jnp.mean(jnp.stack(extract_augmented_text_features(t_prompt), axis=0), axis=0)
    return avg_t_feature - avg_r_feature

def compute_text_features(t_prompt):
    '''
    Args:
        t_prompt: str                  | Target prompt for manipulation. 
    
    Return:
        feature: (512,) np.array | Difference of averaged text embeddings in CLIP space.
    '''
    avg_t_feature = jnp.mean(jnp.stack(extract_augmented_text_features(t_prompt), axis=0), axis=0)
    return avg_t_feature