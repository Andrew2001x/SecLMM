from PIL import Image
import requests
from transformers import AutoProcessor, FlaxCLIPModel, CLIPConfig
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

# Get the PID of the current Python process
pid = os.getpid()
print(f"The PID of the current Python process is: {pid}")

import time

# Start timing the execution
start = time.time()

# Load the CIFAR-10 test dataset
cifar_10_test = load_dataset('cifar10', split='test')
labels = cifar_10_test.features['label'].names

# Create label dictionaries
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt template for CLIP text encoding
prompt = [f"a photo of a {label}" for label in labels]

# Load the pre-trained CLIP model and processor
model = FlaxCLIPModel.from_pretrained("clip")
processor = AutoProcessor.from_pretrained("clip")

# Define a function to predict labels for a batch of images
def get_image_predict_label(images):
    # Preprocess images and text prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    # Perform forward pass through the CLIP model
    outputs = model(**inputs)
    # Extract logits per image (similarity scores between images and text prompts)
    logits_per_image = outputs.logits_per_image
    # Compute probabilities using softmax
    probs = jax.nn.softmax(logits_per_image, axis=1)
    # Get predicted label IDs by selecting the class with the highest probability
    label_ids = jnp.argmax(probs, axis=1).tolist()
    # Map label IDs to label names
    return [id_label_dict[label_id] for label_id in label_ids]

# Test the prediction function on a small subset of images
images = [_['img'] for _ in cifar_10_test.select(range(1))]
test_labels = get_image_predict_label(images=images)

# Print the predicted labels
print(test_labels)

# End timing the execution
end = time.time()
execution_time = end - start

# Print the execution time
print(execution_time)
