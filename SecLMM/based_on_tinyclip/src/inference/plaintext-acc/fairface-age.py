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

cifar_10_test = load_dataset('fairface', split='validation')
labels = cifar_10_test.features['age'].names
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

prompt = ['a photo of people between the ages of 0 and 2','a photo of people between the ages of 3 and 9', 'a photo of people between the ages of 10 and 19', 'a photo of people between the ages of 20 and 29', 'a photo of people between the ages of 30 and 39', 'a photo of people between the ages of 40 and 49', 'a photo of people between the ages of 50 and 59',  'a photo of people between the ages of 60 and 69','a photo of people aged 70 or older' ]

model = FlaxCLIPModel.from_pretrained("clip")
processor = AutoProcessor.from_pretrained("clip")
def get_image_predict_label(images):
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = jax.nn.softmax(logits_per_image, axis=1)
    label_ids = jnp.argmax(probs, axis=1).tolist()
    return [id_label_dict[label_id] for label_id in label_ids]
images = [_['image'] for _ in cifar_10_test.select(range(5))]
test_labels = get_image_predict_label(images=images)

from sklearn.metrics import classification_report
from tqdm import tqdm
import time

y_true = []
y_pred = []

batch_size = 300
start = 0
end = batch_size
while start < len(cifar_10_test):
    sample = cifar_10_test[start:end]
    img_list, label_id_list = sample['image'], sample['age']
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list])
    y_pred.extend(get_image_predict_label(images=img_list))
    start = end
    end += batch_size
    print(start, end)

print(classification_report(y_true, y_pred, target_names=labels, digits=4))






