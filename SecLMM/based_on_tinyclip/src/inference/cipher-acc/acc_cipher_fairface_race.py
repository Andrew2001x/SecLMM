
from datasets import load_dataset
import numpy as np
from tqdm import trange
import joblib
import secretflow as sf
from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
import jax
import jax.numpy as jnp
import argparse
import spu.utils.distributed as ppd
import spu.intrinsic as intrinsic
import spu.spu_pb2 as spu_pb2
from contextlib import contextmanager
from transformers import AutoProcessor, FlaxCLIPModel, CLIPConfig
import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import EvalPrediction
import requests
from PIL import Image
from sklearn.metrics import matthews_corrcoef
from transformers.image_processing_utils import BatchFeature

def classifier(input_ids, attention_mask, pixel_values, params):
    config = CLIPConfig()
    model = FlaxCLIPModel(config=config)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, params=params)
    logits_per_image = outputs.logits_per_image
    probs = jax.nn.softmax(logits_per_image, axis=1)
    label_ids = jnp.argmax(probs, axis=1)
    return label_ids

sf.shutdown()

sf.init(['alice', 'dave'], address='local')

conf = sf.utils.testing.cluster_def(parties=['alice', 'dave'])
conf['runtime_config']['protocol'] = 'SEMI2K'
conf['runtime_config']['field'] = 'FM128'
conf['runtime_config']['enable_pphlo_profile'] = True
conf['runtime_config']['enable_hal_profile'] = True
conf['runtime_config']['enable_pphlo_trace'] = False
conf['runtime_config']['enable_action_trace'] = False
conf['runtime_config']['fxpFractionBits'] = 36
conf['runtime_config']['fxpDivGoldschmidtIters'] = 3
conf['runtime_config']['fxpExpMode'] = 1

spu = sf.SPU(conf)

alice, dave = sf.PYU('alice'), sf.PYU('dave')




def get_token_ids1():
    with open('data/prompt_fairface_race.txt', 'r') as file:
        content = file.read()

    prompt= np.array(ast.literal_eval(content), dtype=int)
    prompt=jnp.array(prompt)
    return prompt
def get_token_ids2():
    with open('data/mask_fairface_race.txt', 'r') as file:
        content = file.read()

    mask= np.array(ast.literal_eval(content), dtype=int)
    mask=jnp.array(mask)
    return mask



def get_token_ids3():
    cifar_10_test = load_dataset('fairface', split='validation')
    images = [_['image'] for _ in cifar_10_test.select(range(50))]
    processor = AutoProcessor.from_pretrained("clip")
    prompt = ['a photo of East Asian group', 'a photo of Indian group', 'a photo of Black group', 'a photo of White group', 'a photo of Middle Eastern group', 'a photo of Latino_Hispanic group', 'a photo of Southeast Asian group']
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    p = inputs.pixel_values
    return p

def get_model_params():
    pretrained_model = FlaxCLIPModel.from_pretrained("clip")
    return pretrained_model.params

model_params = alice(get_model_params)()
input_ids_s = dave(get_token_ids1)()
attention_mask_s = dave(get_token_ids2)()
pixel_values_s = dave(get_token_ids3)()

device = spu
model_params_, input_ids_s, attention_mask_s, pixel_values_s = model_params.to(device), input_ids_s.to(device), attention_mask_s.to(device), pixel_values_s.to(device)

output_token_ids = spu(classifier)(
    input_ids_s, attention_mask_s, pixel_values_s, model_params_
)

outputs_ids = sf.reveal(output_token_ids)
print(outputs_ids)
