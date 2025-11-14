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

# Initialize SecretFlow with parties 'alice', 'bob', 'carol', and 'dave'
sf.init(['alice', 'bob', 'carol', 'dave'], address='local')

# Define the cluster configuration for SecretFlow
conf = sf.utils.testing.cluster_def(parties=['alice', 'bob', 'carol'])
conf['runtime_config']['protocol'] = 'SEMI2K'  # Set protocol to SEMI2K
conf['runtime_config']['field'] = 'FM128'  # Set field to FM128
conf['runtime_config']['enable_pphlo_profile'] = True  # Enable PPHLO profiling
conf['runtime_config']['enable_hal_profile'] = True  # Enable HAL profiling
conf['runtime_config']['enable_pphlo_trace'] = False  # Disable PPHLO tracing
conf['runtime_config']['enable_action_trace'] = False  # Disable action tracing
conf['runtime_config']['fxpFractionBits'] = 36  # Set fixed-point fraction bits
conf['runtime_config']['fxpDivGoldschmidtIters'] = 3  # Set Goldschmidt iterations for division
conf['runtime_config']['fxpExpMode']极1  # Set fixed-point exponent mode

# Initialize SPU (Secure Processing Unit) with the configuration
spu = sf.SPU(conf)

# Define PYU (Private Yielding Unit) instances for 'alice' and 'dave'
alice, dave = sf.PYU('alice'), sf.PYU('dave')

# Function to get token IDs from a file
def get_token_ids1():
    with open('prompt_mask/prompt_tinyimagenet.txt', 'r') as file:
        content = file.read()

    prompt= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    prompt=jnp.array(prompt)  # Convert to JAX array
    return prompt

# Function to get attention mask from a file
def极get_token_ids2():
    with open('prompt_mask/mask_tinyimagenet.txt', 'r') as file:
        content = file.read()

    mask= np.array(ast.literal_eval(content), dtype=int)  # Convert content to numpy array
    mask=jnp.array(mask)  # Convert to JAX array
    return mask

# Function to get pixel values from the TinyImageNet dataset
def get_token_ids3():
    ti_validate = load_dataset('tinyimage', split='validation')  # Load TinyImageNet validation dataset
    images = [_['image'] for _ in ti_validate.select(range(50))]  # Select first 50 images
    processor = AutoProcessor.from_pretrained("clip")  # Load CLIP processor
    prompt = ['a photo of a goldfish', 'a photo of a fire salamander', 'a photo of a American bullfrog', 'a photo of a tailed frog', 'a photo of a American alligator', 'a photo of a boa constrictor', 'a photo of a trilobite', 'a photo of a scorpion', 'a photo of a southern black widow', 'a photo of a tarantula', 'a photo of a centipede', 'a photo of a koala', 'a photo of a jellyfish', 'a photo极of a brain coral', 'a photo of a snail', 'a photo of a sea slug', 'a photo of a American lobster', 'a photo of a spiny lobster', 'a photo of a black stork', 'a photo of a king penguin', 'a photo of a albatross', 'a photo of a dugong', 'a photo of a Yorkshire Terrier', 'a photo of a Golden Retriever', 'a photo of a Labrador Retriever', 'a photo of a German Shepherd Dog', 'a photo of a Standard Poodle', 'a photo of a tabby cat', 'a photo of a Persian cat', 'a photo of a Egyptian Mau', 'a photo of a cougar', 'a photo of a lion', 'a photo of a brown bear', 'a photo of a ladybug', 'a photo of a grasshopper', 'a photo of a stick insect', 'a photo of a cockroach', 'a photo of a praying mantis', 'a photo of a dragonfly', 'a photo of a monarch butterfly', 'a photo of a sulphur butterfly', 'a photo of a sea cucumber', 'a photo of a guinea pig', 'a photo of a pig', 'a photo of a ox', 'a photo of a bison', 'a photo of a bighorn sheep', 'a photo of a gazelle', 'a photo of a arabian camel', 'a photo of a orangutan', 'a photo of a chimpanzee', 'a photo of a baboon', 'a photo of a African bush elephant', 'a photo of a red panda', 'a photo of a abacus', 'a photo of a academic gown', 'a photo of a altar', 'a photo of a backpack', 'a photo of a baluster / handrail', 'a photo of a barbershop', 'a photo of a barn', 'a photo of a barrel', 'a photo of a basketball', 'a photo of a bathtub', 'a photo of a station wagon', 'a photo of a lighthouse', 'a photo of a beaker', 'a photo of a beer bottle', 'a photo of a bikini', 'a photo of a binoculars', 'a photo of a birdhouse', '极a photo of a bow tie', 'a photo of a brass memorial plaque', 'a photo极of a bucket', 'a photo of a high-speed train', 'a photo of a butcher shop', 'a photo of a candle', 'a photo of a cannon', 'a photo of a cardigan', 'a photo of a automated teller machine', 'a photo of a CD player', 'a photo of a storage chest', 'a photo of a Christmas stocking', 'a photo of a cliff dwelling', 'a photo of a computer keyboard', 'a photo of a candy store', 'a photo of a convertible', 'a photo of a crane bird', 'a photo of a dam', 'a photo of a desk', 'a photo of a dining table', 'a photo of a dumbbell', 'a photo of a flagpole', 'a photo of a fly', 'a photo of a fountain', 'a photo of a freight car', 'a photo of a frying pan', 'a photo of a fur coat', 'a photo of a gas mask or respirator', 'a photo of a go-kart', 'a photo of a gondola', 'a photo of a hourglass', 'a photo of a iPod', 'a photo of a rickshaw', 'a photo of a kimono', 'a photo of a lampshade', 'a photo of a lawn mower', 'a photo of a lifeboat', 'a photo of a limousine', 'a photo of a magnetic compass', 'a photo of a maypole', 'a photo of a military uniform', 'a photo of a miniskirt', 'a photo of a moving van', 'a photo of a neck brace', 'a photo of a obelisk', 'a photo of a oboe', 'a photo of a pipe organ', 'a photo of a parking meter', 'a photo of a payphone', 'a photo of a picket fence', 'a photo of a pill bottle', 'a photo of a plunger', 'a photo of a police van', 'a photo of a poncho', 'a photo of a soda bottle', "a photo of a potter's wheel", 'a photo of a missile', 'a photo of a punching bag', 'a photo of a refrigerator', 'a photo of a remote control', 'a photo of a rocking chair', 'a photo of a rugby ball', 'a photo of a sandal', 'a photo of a school bus', 'a photo of a scoreboard', 'a photo of a sewing machine', 'a photo of a snorkel', 'a photo of a sock', 'a photo of a sombrero', 'a photo of a space heater', 'a photo of a spider web', 'a photo of a sports car', 'a photo of a through arch bridge', 'a photo of a stopwatch', 'a photo of a sunglasses', 'a photo of a suspension bridge', 'a photo of a swim trunks / shorts', 'a photo of a syringe', 'a photo of a teapot', 'a photo of a teddy bear', 'a photo of a thatched roof', 'a photo of a torch', 'a photo of a tractor', 'a photo of a triumphal arch', 'a photo of a trolleybus', 'a photo of a turnstile', 'a photo of a umbrella', 'a photo of a vestment', 'a photo of a viaduct', 'a photo of a volleyball', 'a photo of a water jug', 'a photo of a water tower', 'a photo of a wok', 'a photo of a wooden spoon', 'a photo of a comic book', 'a photo of a fishing casting reel', 'a photo of a guacamole', 'a photo of a ice cream', 'a photo of a popsicle', 'a photo of a goose', 'a photo of a drumstick', 'a photo of a plate', 'a photo of a pretzel', 'a photo of a mashed potatoes', 'a photo of a cauliflower', 'a photo of a bell pepper', 'a photo of a lemon', 'a photo of a banana', 'a photo of a pomegranate', 'a photo of a meatloaf', 'a photo of a pizza', 'a photo of a pot pie', 'a photo of a espresso', 'a photo of a bee', 'a photo of a apron', 'a photo of a pole', 'a photo of a Chihuahua', 'a photo of a mountain', 'a photo of a cliff', 'a photo of a coral reef', 'a photo of a lakeshore', 'a photo of a beach', 'a photo of a acorn', 'a photo of a broom', 'a photo of a mushroom', 'a photo of a metal nail', 'a photo of a chain', 'a photo of a slug', 'a photo of a orange']  # Define prompts
    inputs = processor(text=prompt, images=images, return_tensors="jax", padding=True)  # Process inputs
    p = inputs.pixel_values  # Get pixel values
    return p

# Function to get pre-trained CLIP model parameters
def get_model_params():
    pretrained_model = FlaxCLIPModel.from_pretrained("clip")  # Load pre-trained CLIP model
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
