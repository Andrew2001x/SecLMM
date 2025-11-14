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
from transformers.image_processing_utils import BatchFeature  # For batch feature processing

# Define a classifier function using CLIP model
def classifier(input_ids, attention_mask, pixel_values, params):
    config = CLIPConfig()  # Initialize CLIP configuration
    model = FlaxCLIPModel(config=config)  # Initialize CLIP model

    # Get model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, params=params)
    logits_per_image = outputs.logits_per_image  # Logits for each image
    probs = jax.nn.softmax(logits_per_image, axis=1)  # Softmax probabilities
    label_ids = jnp.argmax(probs, axis=1)  # Predicted label IDs
    return label_ids

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
    with open('prompt_mask/prompt_cifar100.txt', 'r') as file:
        content = file.read()

    prompt= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    prompt=jnp.array(prompt)  # Convert to JAX array
    return prompt

# Function to get attention mask from a file
def get_token_ids2():
    with open('prompt_mask/mask_cifar100.txt', 'r') as file:
        content = file.read()

    mask= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    mask=jnp.array(mask)  # Convert to JAX array
    return mask

# Function to get pixel values from the CIFAR-100 dataset
def get_token_ids3():
    cifar_100_test = load_dataset('cifar100', split='test')  # Load CIFAR-100 test dataset
    images = [_['img'] for _ in cifar_100_test.select(range(50))]  # Select first 50 images
    processor = AutoProcessor.from_pretrained("clipq")  # Load CLIP processor
    prompt = ['a photo of a apple', 'a photo of a aquarium_fish', 'a photo of a baby', 'a photo of a bear', 'a photo of a beaver', 'a photo of a bed', 'a photo of a bee', 'a photo of a beetle', 'a photo of a bicycle', 'a photo of a bottle', 'a photo of a bowl', 'a photo of a boy', 'a photo of a bridge', 'a photo of a bus', 'a photo of a butterfly', 'a photo of a camel', 'a photo of a can', 'a photo of a castle', 'a photo of a caterpillar', 'a photo of a cattle', 'a photo of a chair', 'a photo of a chimpanzee', 'a photo of a clock', 'a photo of a cloud', 'a photo of a cockroach', 'a photo of a couch', 'a photo of a cra', 'a photo of a crocodile', 'a photo of a cup', 'a photo of a dinosaur', 'a photo of a dolphin', 'a photo of a elephant', 'a photo of a flatfish', 'a photo of a forest', 'a photo of a fox', 'a photo of a girl', 'a photo of a hamster', 'a photo of a house', 'a photo of a kangaroo', 'a photo of a keyboard', 'a photo of a lamp', 'a photo of a lawn_mower', 'a photo of a leopard', 'a photo of a lion', 'a photo of a lizard', 'a photo of a lobster', 'a photo of a man', 'a photo of a maple_tree', 'a photo of a motorcycle', 'a photo of a mountain', 'a photo of a mouse', 'a photo of a mushroom', 'a photo of a oak_tree', 'a photo of a orange', 'a photo of a orchid', 'a photo of a otter', 'a photo of a palm_tree', 'a photo of a pear', 'a photo of a pickup_truck', 'a photo of a pine_tree', 'a photo of a plain', 'a photo of a plate', 'a photo of a poppy', 'a photo of a porcupine', 'a photo of a possum', 'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a ray', 'a photo of a road', 'a photo of a rocket', 'a photo of a rose', 'a photo of a sea', 'a photo of a seal', 'a photo of a shark', 'a photo of a shrew', 'a photo of a skunk', 'a photo of a skyscraper', 'a photo of a snail', 'a photo of a snake', 'a photo of a spider', 'a photo of a squirrel', 'a photo of a streetcar', 'a photo of a sunflower', 'a photo of a sweet_pepper', 'a photo of a table', 'a photo of a tank', 'a photo of a telephone', 'a photo of a television', 'a photo of a tiger', 'a photo of a tractor', 'a photo of a train', 'a photo of a trout', 'a photo of a tulip', 'a photo of a turtle', 'a photo of a wardrobe', 'a photo of a whale', 'a photo of a willow_tree', 'a photo of a wolf', 'a photo of a woman', 'a photo of a worm']  # Define prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)  # Process inputs
    p = inputs.pixel_values  # Get pixel values
    return p

# Function to get pre-trained CLIP model parameters
def get_model_params():
    pretrained_model = FlaxCLIPModel.from_pretrained("clipq")  # Load pre-trained CLIP model
    return pretrained_model.params  # Return model parameters

# Get model parameters and token IDs using PYU instances
model_params = alice(get_model_params)()
input_ids_s = dave(get_token_ids1)()
attention_mask_s = dave(get_token_ids2)()
pixel_values_s = dave(get_token_ids3)()

# Move data to the encryted environment
device = spu
model_params_, input_ids_s, attention_mask_s, pixel_values_s = model_params.to(device), input_ids_s.to(device), attention_mask_s.to(device), pixel_values_s.to(device)

# Run the secure inference
output_token_ids = spu(classifier)(
    input_ids_s, attention_mask_s, pixel_values_s, model_params_
)

# Reveal the output token IDs
outputs_ids = sf.reveal(output_token_ids)
print(outputs_ids)  # Print the output IDs
