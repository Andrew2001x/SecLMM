from PIL import Image
import requests
from transformers import AutoProcessor, FlaxCLIPModel,CLIPConfig
from datasets import load_dataset
import numpy as np
from tqdm import trange
from typing import Any, Callable, Dict, Optional, Tuple, Union
import jax.nn as jnn
import flax.linen as nn
from flax.linen.linear import Array
import jax
import jax.numpy as jnp
import argparse
import torch
import datasets
from datasets import load_metric
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from itertools import islice
import jax.nn as jnn

from datasets import load_dataset
import os

# 获取当前进程的 PID
pid = os.getpid()

print(f"当前 Python 进程的 PID 是：{pid}")

import time

start=time.time()
cifar_10_test = load_dataset('cifar10', split='test')
labels = cifar_10_test.features['label'].names
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))
prompt = [f"a photo of a {label}" for label in labels]
model = FlaxCLIPModel.from_pretrained("clipq")
processor = AutoProcessor.from_pretrained("clipq")
def get_image_predict_label(images):
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = jax.nn.softmax(logits_per_image, axis=1)
    label_ids = jnp.argmax(probs, axis=1).tolist()
    return [id_label_dict[label_id] for label_id in label_ids]
images = [_['img'] for _ in cifar_10_test.select(range(1))]
test_labels = get_image_predict_label(images=images)

print(test_labels)
end=time.time()
aaa=end-start
print(aaa)