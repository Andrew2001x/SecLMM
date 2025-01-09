
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
    i=jnp.array([[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 271, 537, 273, 49407, 49407, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 274, 537, 280, 49407, 49407, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 272, 271, 537, 272, 280, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 273, 271, 537, 273, 280, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 274, 271, 537, 274, 280, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 275, 271, 537, 275, 280, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 276, 271, 537, 276, 280, 49407],
[49406, 320, 1125, 539, 1047, 1957, 518, 2321, 539, 277, 271, 537, 277, 280, 49407],
[49406, 320, 1125, 539, 1047, 3889, 278, 271, 541, 7700, 49407, 49407, 49407, 49407, 49407]]

)
    return i
def get_token_ids2():
    a=jnp.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]


)
    return a
def get_token_ids3():
    cifar_10_test = load_dataset('fairface', split='validation')
    images = [_['image'] for _ in cifar_10_test.select(range(50))]
    processor = AutoProcessor.from_pretrained("clip")
    prompt = ['a photo of people between the ages of 0 and 2','a photo of people between the ages of 3 and 9', 'a photo of people between the ages of 10 and 19', 'a photo of people between the ages of 20 and 29', 'a photo of people between the ages of 30 and 39', 'a photo of people between the ages of 40 and 49', 'a photo of people between the ages of 50 and 59',  'a photo of people between the ages of 60 and 69','a photo of people aged 70 or older' ]
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
