import torch
from PIL import Image
import open_clip
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import random
from datasets import load_dataset
from torchvision import transforms

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
np.random.seed(seed)
random.seed(seed)

# Enable deterministic algorithms in CuDNN (for GPUs)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and preprocessing transforms
arch = 'TinyCLIP-ViT-39M-16-Text-19M'

# Create the model and move it to the GPU
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='/na/mq/dist/mo/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt')
model.to(device)  # Move the model to the GPU

# Ensure the model is in evaluation mode
model.eval()

# Load the dataset
dataset = load_dataset('out3', split='validation')  # Load the FairFace validation set
labels = dataset.features['gender'].names
prompt = ['a photo of a male', 'a photo of a female']

# Set up the tokenizer for text encoding
tokenizer = open_clip.get_tokenizer(arch)

# Create label dictionaries
label_id_dict = {label: idx for idx, label in enumerate(labels)}
id_label_dict = {idx: label for idx, label in enumerate(labels)}

# Extract image and label fields from the dataset
img_list = dataset['image']  # Image field
label_id_list = dataset['gender']  # Label field

# Define a function for batch prediction
def get_image_predict_label(images):
    # Preprocess the images
    inputs = [preprocess(image) for image in images]
    inputs = torch.stack(inputs).to(device)  # Move the data to the GPU
    
    # Perform inference
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        
        # Tokenize the text descriptions
        text_tokens = tokenizer(prompt).to(device)  # Move the text input to the GPU
        text_features = model.encode_text(text_tokens)
        
        # Normalize image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity between image and text features
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get the predicted label indices
        _, predicted_label_index = text_probs.max(dim=1)
        return [id_label_dict[label_id.item()] for label_id in predicted_label_index]

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Set batch size
batch_size = 300
total_samples = len(dataset)

# Iterate through the dataset in batches
for start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    end = min(start + batch_size, total_samples)  # Ensure the batch does not exceed the dataset size
    sample = dataset[start:end]
    img_list_batch = sample['image']
    label_id_list_batch = sample['gender']
    
    # Collect true labels
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list_batch])
    
    # Get predicted labels
    y_pred.extend(get_image_predict_label(images=img_list_batch))

# Print the classification report
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
