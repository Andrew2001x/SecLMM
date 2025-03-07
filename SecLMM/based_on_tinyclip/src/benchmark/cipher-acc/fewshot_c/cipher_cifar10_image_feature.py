# Import necessary libraries
from datasets import load_dataset  # For loading datasets
import numpy as np  # For numerical operations
from tqdm import trange  # For progress bars
import joblib  # For saving/loading Python objects
import secretflow as sf  # For secure multi-party computation
from typing import Any, Callable, Dict, Optional, Tuple, Union  # For type hints
import jax.nn as jnn  # JAX neural network module
import flax.linen as nn  # Flax neural network module
from flax.linen.linear import Array  # For array operations in Flax
import jax  # For JAX operations
import jax.numpy as jnp  # JAX version of NumPy
import argparse  # For parsing command-line arguments
import spu.utils.distributed as ppd  # For distributed computing in SPU
import spu.intrinsic as intrinsic  # For intrinsic functions in SPU
import spu.spu_pb2 as spu_pb2  # For SPU protocol buffer definitions
from contextlib import contextmanager  # For context management
from transformers import AutoProcessor, FlaxCLIPModel, CLIPConfig  # For CLIP model and processor
import torch  # PyTorch library
from torch.utils.data import Dataset  # For custom datasets in PyTorch
import numpy as np  # For numerical operations
from transformers import EvalPrediction  # For evaluation predictions
import requests  # For HTTP requests
from PIL import Image  # For image processing
from sklearn.metrics import matthews_corrcoef  # For Matthews correlation coefficient
from transformers.image极processing_utils import BatchFeature  # For batch feature processing
import ast  # For safely evaluating strings as Python expressions

# Define a classifier function using CLIP model
def classifier(input_ids, attention_mask, pixel_values, params):
    config = CLIPConfig()  # Initialize CLIP configuration
    model = FlaxCLIPModel(config=config)  # Initialize CLIP model

    # Get model outputs
    outputs = model.get_image_features(pixel_values=pixel_values, attention_mask=attention_mask, params=params)
    return outputs 

# Shutdown SecretFlow to reset any existing configurations
sf.shutdown()

# Initialize SecretFlow with parties 'alice' and 'dave'
sf.init(['alice', 'dave'], address='local')

# Define the cluster configuration for SecretFlow
conf = sf.utils.testing.cluster_def(parties=['alice', 'dave'])
conf['runtime_config']['protocol'] = 'SEMI2K'  # Set protocol to SEMI2K
conf['runtime_config']['field'] = 'FM128'  # Set field to FM128
conf['runtime_config']['enable_pphlo_profile'] = True  # Enable PPHLO profiling
conf['runtime_config']['enable_hal_profile'] = True  # Enable HAL profiling
conf['runtime_config']['enable_pphlo_trace'] = False  # Disable PPHLO tracing
conf['runtime_config']['enable_action_trace'] = False  # Disable action tracing
conf['runtime_config']['fxpFractionBits'] = 36  # Set fixed-point fraction bits
conf['runtime_config']['fxpDivGoldschmidtIters'] = 3  # Set Goldschmidt iterations for division
conf['runtime_config']['fxpExpMode'] = 1  # Set fixed-point exponent mode

# Initialize SPU (Secure Processing Unit) with the configuration
spu = sf.SPU(conf)

# Define PYU (Private Yielding Unit) instances for 'alice' and 'dave'
alice, dave = sf.PYU('alice'), sf.PYU('dave')

# Function to get token IDs from a file
def get_token_ids1():
    with open('prompt_mask/prompt_cifar10.txt', 'r') as file:
        content = file.read()

    prompt= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    prompt=jnp.array(prompt)  # Convert to JAX array
    return prompt

# Function to get attention mask from a file
def get_token_ids2():
    with open('prompt_mask/mask_cifar10.txt', 'r') as file:
        content = file.read()

    mask= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    mask=jnp.array(mask)  # Convert to JAX array
    return mask

# Function to get pixel values from the CIFAR-10 dataset
def get_token_ids3():
    cifar_10_test = load_dataset('out', split='test')  # Load CIFAR-10 test dataset
    images = [_['img'] for _ in cifar_10_test.select(range(10))]  # Select first 10 images
    processor = AutoProcessor.from_pretrained("b")  # Load CLIP processor
    prompt = ['a photo of a airplane', 'a photo of a automobile', 'a photo of a bird', 'a photo of a cat', 'a photo of a deer', 'a photo of a dog', 'a photo of a frog', 'a photo of a horse', 'a photo of a ship', 'a photo of a truck']  # Define prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)  # Process inputs
    p = inputs.pixel_values  # Get pixel values
    return p

# Function to get pre-trained CLIP model parameters
def get_model_params():
    pretrained_model = FlaxCLIPModel.from_pretrained("b")  # Load pre-trained CLIP model
    return pretrained_model.params  # Return model parameters

# Get model parameters and token IDs using PYU instances
model_params = alice(get_model_params)()
input_ids_s = dave(get_token_ids1)()
attention_mask_s = dave(get_token_ids2)()
pixel_values_s = dave(get_token_ids3)()

# Move data to the encrypted environment
device = spu
model_params_, input_ids_s, attention_mask_s, pixel_values_s = model_params.to(device), input_ids_s.to(device), attention_mask_s.to(device), pixel_values_s.to(device)

# Run the secure inference
output_token_ids = spu(classifier)(
    input_ids_s, attention_mask_s, pixel_values_s, model_params_
)

# Reveal the output token IDs
outputs_ids = sf.reveal(output_token_ids)
print(outputs_ids)  # Print the output IDs
np.savetxt("outputs_image_cifar10.txt", outputs_ids, fmt="%.8f")