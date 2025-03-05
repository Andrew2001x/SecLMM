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

# Load the FairFace validation dataset
cifar_10_test = load_dataset('fairface', split='validation')
# Extract race labels
labels = cifar_10_test.features['race'].names
# Create mappings between race labels and their corresponding IDs
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt template for CLIP text encoding
prompt = [
    'a photo of East Asian group',
    'a photo of Indian group',
    'a photo of Black group',
    'a photo of White group',
    'a photo of Middle Eastern group',
    'a photo of Latino_Hispanic group',
    'a photo of Southeast Asian group'
]

# Load the pre-trained CLIP model and processor
model = FlaxCLIPModel.from_pretrained("clip")
processor = AutoProcessor.from_pretrained("clip")

# Define a function to predict race labels for a batch of images
def get_image_predict_label(images):
    # Preprocess images and text prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    # Perform forward pass through the CLIP model
    outputs = model(**inputs)
    # Extract logits per image (similarity scores between images and text prompts)
    logits_per_image = outputs.logits_per_image
    # Compute probabilities using softmax
    probs = jax.nn.softmax(logits_per_image, axis=1)
    # Get predicted label IDs by selecting the race with the highest probability
    label_ids = jnp.argmax(probs, axis=1).tolist()
    # Map label IDs to race labels
    return [id_label_dict[label_id] for label_id in label_ids]

# Test the prediction function on a small subset of images
images = [_['image'] for _ in cifar_10_test.select(range(5))]
test_labels = get_image_predict_label(images=images)

# Import necessary libraries for evaluation
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Define batch size for processing the dataset
batch_size = 300
start = 0
end = batch_size

# Iterate through the dataset in batches
while start < len(cifar_10_test):
    # Extract a batch of images and their corresponding race labels
    sample = cifar_10_test[start:end]
    img_list, label_id_list = sample['image'], sample['race']
    # Map true label IDs to race labels and store them
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list])
    # Predict race labels for the batch of images
    y_pred.extend(get_image_predict_label(images=img_list))
    # Update batch indices for the next iteration
    start = end
    end += batch_size
    print(f"Processed samples from {start} to {end}")

# Generate and print the classification report
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
