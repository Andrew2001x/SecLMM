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

# Define the classifier function
def classifier(input_ids, attention_mask, pixel_values, params):
    # Initialize the CLIP model with the given configuration
    config = CLIPConfig()
    model = FlaxCLIPModel(config=config)

    # Perform forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, params=params)
    # Extract logits per image (similarity scores between images and text prompts)
    logits_per_image = outputs.logits_per_image
    # Compute probabilities using softmax
    probs = jax.nn.softmax(logits_per_image, axis=1)
    # Get the predicted label indices
    label_ids = jnp.argmax(probs, axis=1)
    return label_ids

# Shutdown any existing SecretFlow sessions
sf.shutdown()

# Initialize SecretFlow with the specified parties
sf.init(['alice', 'dave'], address='local')

# Configure the SecretFlow cluster
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

# Initialize the SPU (Secure Processing Unit) with the given configuration
spu = sf.SPU(conf)

# Define the PYU (Private Yielding Unit) instances for Alice and Dave
alice, dave = sf.PYU('alice'), sf.PYU('dave')

# Define a function to load token IDs from a file
def get_token_ids1():
    with open('prompt_mask/prompt_fairface_race.txt', 'r') as file:
        content = file.read()

    # Convert the content to a NumPy array and then to a JAX array
    prompt = np.array(ast.literal_eval(content), dtype=int)
    prompt = jnp.array(prompt)
    return prompt

# Define a function to load attention masks from a file
def get_token_ids2():
    with open('prompt_mask/mask_fairface_race.txt', 'r') as file:
        content = file.read()

    # Convert the content to a NumPy array and then to a JAX array
    mask = np.array(ast.literal_eval(content), dtype=int)
    mask = jnp.array(mask)
    return mask

# Define a function to load pixel values from the dataset
def get_token_ids3():
    # Load the FairFace validation dataset
    cifar_10_test = load_dataset('fairface', split='validation')
    # Extract images from the dataset
    images = [_['image'] for _ in cifar_10_test.select(range(50))]
    # Initialize the processor for the CLIP model
    processor = AutoProcessor.from_pretrained("clipq")
    # Define the text prompts
    prompt = [
        'a photo of East Asian group', 'a photo of Indian group', 'a photo of Black group',
        'a photo of White group', 'a photo of Middle Eastern group', 'a photo of Latino_Hispanic group',
        'a photo of Southeast Asian group'
    ]
    # Preprocess the images and text prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)
    # Extract the pixel values
    p = inputs.pixel_values
    return p

# Define a function to load the model parameters
def get_model_params():
    # Load the pre-trained CLIP model
    pretrained_model = FlaxCLIPModel.from_pretrained("clipq")
    return pretrained_model.params

# Load the model parameters and token IDs using the PYU instances
model_params = alice(get_model_params)()
input_ids_s = dave(get_token_ids1)()
attention_mask_s = dave(get_token_ids2)()
pixel_values_s = dave(get_token_ids3)()

# Move the model parameters and token IDs to the encrypted environment
device = spu
model_params_, input_ids_s, attention_mask_s, pixel_values_s = model_params.to(device), input_ids_s.to(device), attention_mask_s.to(device), pixel_values_s.to(device)

# Perform secure inference 
output_token_ids = spu(classifier)(
    input_ids_s, attention_mask_s, pixel_values_s, model_params_
)

# Reveal the output token IDs
outputs_ids = sf.reveal(output_token_ids)
print(outputs_ids)
