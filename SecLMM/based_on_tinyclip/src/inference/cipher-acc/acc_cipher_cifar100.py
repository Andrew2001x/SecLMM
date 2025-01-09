
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
    i=jnp.array([[49406,320,1125,539,320,3055,49407,49407,49407],[49406,320,1125,539,320,16814,318,2759,49407],[49406,320,1125,539,320,1794,49407,49407,49407],[49406,320,1125,539,320,4298,49407,49407,49407],[49406,320,1125,539,320,22874,49407,49407,49407],[49406,320,1125,539,320,2722,49407,49407,49407],[49406,320,1125,539,320,5028,49407,49407,49407],[49406,320,1125,539,320,16534,49407,49407,49407],[49406,320,1125,539,320,11652,49407,49407,49407],[49406,320,1125,539,320,5392,49407,49407,49407],[49406,320,1125,539,320,3814,49407,49407,49407],[49406,320,1125,539,320,1876,49407,49407,49407],[49406,320,1125,539,320,2465,49407,49407,49407],[49406,320,1125,539,320,2840,49407,49407,49407],[49406,320,1125,539,320,9738,49407,49407,49407],[49406,320,1125,539,320,21914,49407,49407,49407],[49406,320,1125,539,320,753,49407,49407,49407],[49406,320,1125,539,320,3540,49407,49407,49407],[49406,320,1125,539,320,27111,49407,49407,49407],[49406,320,1125,539,320,13644,49407,49407,49407],[49406,320,1125,539,320,4269,49407,49407,49407],[49406,320,1125,539,320,10543,1072,14080,49407],[49406,320,1125,539,320,6716,49407,49407,49407],[49406,320,1125,539,320,3887,49407,49407,49407],[49406,320,1125,539,320,622,916,31073,49407],[49406,320,1125,539,320,12724,49407,49407,49407],[49406,320,1125,539,320,18362,49407,49407,49407],[49406,320,1125,539,320,24757,49407,49407,49407],[49406,320,1125,539,320,1937,49407,49407,49407],[49406,320,1125,539,320,15095,49407,49407,49407],[49406,320,1125,539,320,16464,49407,49407,49407],[49406,320,1125,539,320,10299,49407,49407,49407],[49406,320,1125,539,320,8986,2759,49407,49407],[49406,320,1125,539,320,4167,49407,49407,49407],[49406,320,1125,539,320,3240,49407,49407,49407],[49406,320,1125,539,320,1611,49407,49407,49407],[49406,320,1125,539,320,33313,49407,49407,49407],[49406,320,1125,539,320,1212,49407,49407,49407],[49406,320,1125,539,320,25513,49407,49407,49407],[49406,320,1125,539,320,13017,49407,49407,49407],[49406,320,1125,539,320,10725,49407,49407,49407],[49406,320,1125,539,320,11024,318,30895,49407],[49406,320,1125,539,320,15931,49407,49407,49407],[49406,320,1125,539,320,5567,49407,49407,49407],[49406,320,1125,539,320,17221,49407,49407,49407],[49406,320,1125,539,320,13793,49407,49407,49407],[49406,320,1125,539,320,786,49407,49407,49407],[49406,320,1125,539,320,10570,318,2677,49407],[49406,320,1125,539,320,10297,49407,49407,49407],[49406,320,1125,539,320,3965,49407,49407,49407],[49406,320,1125,539,320,9301,49407,49407,49407],[49406,320,1125,539,320,13011,49407,49407,49407],[49406,320,1125,539,320,7221,318,2677,49407],[49406,320,1125,539,320,4287,49407,49407,49407],[49406,320,1125,539,320,18678,49407,49407,49407],[49406,320,1125,539,320,22456,49407,49407,49407],[49406,320,1125,539,320,8612,318,2677,49407],[49406,320,1125,539,320,18820,49407,49407,49407],[49406,320,1125,539,320,15382,318,4629,49407],[49406,320,1125,539,320,7374,318,2677,49407],[49406,320,1125,539,320,10709,49407,49407,49407],[49406,320,1125,539,320,5135,49407,49407,49407],[49406,320,1125,539,320,15447,49407,49407,49407],[49406,320,1125,539,320,817,5059,715,49407],[49406,320,1125,539,320,38575,49407,49407,49407],[49406,320,1125,539,320,10274,49407,49407,49407],[49406,320,1125,539,320,29516,49407,49407,49407],[49406,320,1125,539,320,3077,49407,49407,49407],[49406,320,1125,539,320,1759,49407,49407,49407],[49406,320,1125,539,320,8383,49407,49407,49407],[49406,320,1125,539,320,3568,49407,49407,49407],[49406,320,1125,539,320,2102,49407,49407,49407],[49406,320,1125,539,320,10159,49407,49407,49407],[49406,320,1125,539,320,7980,49407,49407,49407],[49406,320,1125,539,320,12101,342,49407,49407],[49406,320,1125,539,320,42194,49407,49407,49407],[49406,320,1125,539,320,3075,11187,1284,49407],[49406,320,1125,539,320,23132,49407,49407,49407],[49406,320,1125,539,320,8798,49407,49407,49407],[49406,320,1125,539,320,7622,49407,49407,49407],[49406,320,1125,539,320,14004,49407,49407,49407],[49406,320,1125,539,320,34268,49407,49407,49407],[49406,320,1125,539,320,21559,49407,49407,49407],[49406,320,1125,539,320,2418,318,8253,49407],[49406,320,1125,539,320,2175,49407,49407,49407],[49406,320,1125,539,320,6172,49407,49407,49407],[49406,320,1125,539,320,17243,49407,49407,49407],[49406,320,1125,539,320,8608,49407,49407,49407],[49406,320,1125,539,320,6531,49407,49407,49407],[49406,320,1125,539,320,14607,49407,49407,49407],[49406,320,1125,539,320,3231,49407,49407,49407],[49406,320,1125,539,320,14853,49407,49407,49407],[49406,320,1125,539,320,28389,49407,49407,49407],[49406,320,1125,539,320,10912,49407,49407,49407],[49406,320,1125,539,320,15020,49407,49407,49407],[49406,320,1125,539,320,11650,49407,49407,49407],[49406,320,1125,539,320,15665,318,2677,49407],[49406,320,1125,539,320,5916,49407,49407,49407],[49406,320,1125,539,320,2308,49407,49407,49407],[49406,320,1125,539,320,10945,49407,49407,49407]])
    return i
def get_token_ids2():
    a=jnp.array([[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0],[1,1,1,1,1,1,1,0,0]]
)
    return a
def get_token_ids3():
    cifar_100_test = load_dataset('cifar100', split='test')
    images = [_['img'] for _ in cifar_100_test.select(range(50))]
    processor = AutoProcessor.from_pretrained("clip")
    prompt = ['a photo of a apple', 'a photo of a aquarium_fish', 'a photo of a baby', 'a photo of a bear', 'a photo of a beaver', 'a photo of a bed', 'a photo of a bee', 'a photo of a beetle', 'a photo of a bicycle', 'a photo of a bottle', 'a photo of a bowl', 'a photo of a boy', 'a photo of a bridge', 'a photo of a bus', 'a photo of a butterfly', 'a photo of a camel', 'a photo of a can', 'a photo of a castle', 'a photo of a caterpillar', 'a photo of a cattle', 'a photo of a chair', 'a photo of a chimpanzee', 'a photo of a clock', 'a photo of a cloud', 'a photo of a cockroach', 'a photo of a couch', 'a photo of a cra', 'a photo of a crocodile', 'a photo of a cup', 'a photo of a dinosaur', 'a photo of a dolphin', 'a photo of a elephant', 'a photo of a flatfish', 'a photo of a forest', 'a photo of a fox', 'a photo of a girl', 'a photo of a hamster', 'a photo of a house', 'a photo of a kangaroo', 'a photo of a keyboard', 'a photo of a lamp', 'a photo of a lawn_mower', 'a photo of a leopard', 'a photo of a lion', 'a photo of a lizard', 'a photo of a lobster', 'a photo of a man', 'a photo of a maple_tree', 'a photo of a motorcycle', 'a photo of a mountain', 'a photo of a mouse', 'a photo of a mushroom', 'a photo of a oak_tree', 'a photo of a orange', 'a photo of a orchid', 'a photo of a otter', 'a photo of a palm_tree', 'a photo of a pear', 'a photo of a pickup_truck', 'a photo of a pine_tree', 'a photo of a plain', 'a photo of a plate', 'a photo of a poppy', 'a photo of a porcupine', 'a photo of a possum', 'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a ray', 'a photo of a road', 'a photo of a rocket', 'a photo of a rose', 'a photo of a sea', 'a photo of a seal', 'a photo of a shark', 'a photo of a shrew', 'a photo of a skunk', 'a photo of a skyscraper', 'a photo of a snail', 'a photo of a snake', 'a photo of a spider', 'a photo of a squirrel', 'a photo of a streetcar', 'a photo of a sunflower', 'a photo of a sweet_pepper', 'a photo of a table', 'a photo of a tank', 'a photo of a telephone', 'a photo of a television', 'a photo of a tiger', 'a photo of a tractor', 'a photo of a train', 'a photo of a trout', 'a photo of a tulip', 'a photo of a turtle', 'a photo of a wardrobe', 'a photo of a whale', 'a photo of a willow_tree', 'a photo of a wolf', 'a photo of a woman', 'a photo of a worm']
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
