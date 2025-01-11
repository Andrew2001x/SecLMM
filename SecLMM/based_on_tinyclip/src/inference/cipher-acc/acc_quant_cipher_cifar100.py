
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
    with open('data/prompt_cifar100.txt', 'r') as file:
        content = file.read()

    prompt= np.array(ast.literal_eval(content), dtype=int)
    prompt=jnp.array(prompt)
    return prompt
def get_token_ids2():
    with open('data/mask_cifar100.txt', 'r') as file:
        content = file.read()

    mask= np.array(ast.literal_eval(content), dtype=int)
    mask=jnp.array(mask)
    return mask

def get_token_ids3():
    cifar_100_test = load_dataset('cifar100', split='test')
    images = [_['img'] for _ in cifar_100_test.select(range(50))]
    processor = AutoProcessor.from_pretrained("clipq")
    prompt = ['a photo of a apple', 'a photo of a aquarium_fish', 'a photo of a baby', 'a photo of a bear', 'a photo of a beaver', 'a photo of a bed', 'a photo of a bee', 'a photo of a beetle', 'a photo of a bicycle', 'a photo of a bottle', 'a photo of a bowl', 'a photo of a boy', 'a photo of a bridge', 'a photo of a bus', 'a photo of a butterfly', 'a photo of a camel', 'a photo of a can', 'a photo of a castle', 'a photo of a caterpillar', 'a photo of a cattle', 'a photo of a chair', 'a photo of a chimpanzee', 'a photo of a clock', 'a photo of a cloud', 'a photo of a cockroach', 'a photo of a couch', 'a photo of a cra', 'a photo of a crocodile', 'a photo of a cup', 'a photo of a dinosaur', 'a photo of a dolphin', 'a photo of a elephant', 'a photo of a flatfish', 'a photo of a forest', 'a photo of a fox', 'a photo of a girl', 'a photo of a hamster', 'a photo of a house', 'a photo of a kangaroo', 'a photo of a keyboard', 'a photo of a lamp', 'a photo of a lawn_mower', 'a photo of a leopard', 'a photo of a lion', 'a photo of a lizard', 'a photo of a lobster', 'a photo of a man', 'a photo of a maple_tree', 'a photo of a motorcycle', 'a photo of a mountain', 'a photo of a mouse', 'a photo of a mushroom', 'a photo of a oak_tree', 'a photo of a orange', 'a photo of a orchid', 'a photo of a otter', 'a photo of a palm_tree', 'a photo of a pear', 'a photo of a pickup_truck', 'a photo of a pine_tree', 'a photo of a plain', 'a photo of a plate', 'a photo of a poppy', 'a photo of a porcupine', 'a photo of a possum', 'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a ray', 'a photo of a road', 'a photo of a rocket', 'a photo of a rose', 'a photo of a sea', 'a photo of a seal', 'a photo of a shark', 'a photo of a shrew', 'a photo of a skunk', 'a photo of a skyscraper', 'a photo of a snail', 'a photo of a snake', 'a photo of a spider', 'a photo of a squirrel', 'a photo of a streetcar', 'a photo of a sunflower', 'a photo of a sweet_pepper', 'a photo of a table', 'a photo of a tank', 'a photo of a telephone', 'a photo of a television', 'a photo of a tiger', 'a photo of a tractor', 'a photo of a train', 'a photo of a trout', 'a photo of a tulip', 'a photo of a turtle', 'a photo of a wardrobe', 'a photo of a whale', 'a photo of a willow_tree', 'a photo of a wolf', 'a photo of a woman', 'a photo of a worm']
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    p = inputs.pixel_values
    return p

def get_model_params():
    pretrained_model = FlaxCLIPModel.from_pretrained("clipq")
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
