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

# Load the model and preprocessing transforms
arch = 'TinyCLIP-ViT-39M-16-Text-19M'
checkpoint_path = '/na/mq/dist/mo/yuanshi.pt'
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='/na/mq/dist/mo/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt')

# Ensure the model is in evaluation mode
model.eval()

# Load the dataset
dataset = load_dataset('out1', split='test')  # Load the CIFAR-100 test set

# Set up text descriptions
tokenizer = open_clip.get_tokenizer(arch)
labels = dataset.features['fine_label'].names
prompt = [
    'a photo of a apple', 'a photo of a aquarium_fish', 'a photo of a baby', 'a photo of a bear', 'a photo of a beaver',
    'a photo of a bed', 'a photo of a bee', 'a photo of a beetle', 'a photo of a bicycle', 'a photo of a bottle',
    'a photo of a bowl', 'a photo of a boy', 'a photo of a bridge', 'a photo of a bus', 'a photo of a butterfly',
    'a photo of a camel', 'a photo of a can', 'a photo of a castle', 'a photo of a caterpillar', 'a photo of a cattle',
    'a photo of a chair', 'a photo of a chimpanzee', 'a photo of a clock', 'a photo of a cloud', 'a photo of a cockroach',
    'a photo of a couch', 'a photo of a cra', 'a photo of a crocodile', 'a photo of a cup', 'a photo of a dinosaur',
    'a photo of a dolphin', 'a photo of a elephant', 'a photo of a flatfish', 'a photo of a forest', 'a photo of a fox',
    'a photo of a girl', 'a photo of a hamster', 'a photo of a house', 'a photo of a kangaroo', 'a photo of a keyboard',
    'a photo of a lamp', 'a photo of a lawn_mower', 'a photo of a leopard', 'a photo of a lion', 'a photo of a lizard',
    'a photo of a lobster', 'a photo of a man', 'a photo of a maple_tree', 'a photo of a motorcycle', 'a photo of a mountain',
    'a photo of a mouse', 'a photo of a mushroom', 'a photo of a oak_tree', 'a photo of a orange', 'a photo of a orchid',
    'a photo of a otter', 'a photo of a palm_tree', 'a photo of a pear', 'a photo of a pickup_truck', 'a photo of a pine_tree',
    'a photo of a plain', 'a photo of a plate', 'a photo of a poppy', 'a photo of a porcupine', 'a photo of a possum',
    'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a ray', 'a photo of a road', 'a photo of a rocket',
    'a photo of a rose', 'a photo of a sea', 'a photo of a seal', 'a photo of a shark', 'a photo of a shrew',
    'a photo of a skunk', 'a photo of a skyscraper', 'a photo of a snail', 'a photo of a snake', 'a photo of a spider',
    'a photo of a squirrel', 'a photo of a streetcar', 'a photo of a sunflower', 'a photo of a sweet_pepper', 'a photo of a table',
    'a photo of a tank', 'a photo of a telephone', 'a photo of a television', 'a photo of a tiger', 'a photo of a tractor',
    'a photo of a train', 'a photo of a trout', 'a photo of a tulip', 'a photo of a turtle', 'a photo of a wardrobe',
    'a photo of a whale', 'a photo of a willow_tree', 'a photo of a wolf', 'a photo of a woman', 'a photo of a worm'
]

# Create label dictionaries
label_id_dict = {label: idx for idx, label in enumerate(labels)}
id_label_dict = {idx: label for idx, label in enumerate(labels)}

# Extract image and label fields from the dataset
img_list = dataset['img']  # Image field
label_id_list = dataset['fine_label']  # Label field

# Define a function for batch prediction
def get_image_predict_label(images):
    # Preprocess the images
    inputs = [preprocess(image) for image in images]
    inputs = torch.stack(inputs)
    
    # Perform inference
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        
        # Tokenize the text descriptions
        text_tokens = tokenizer(prompt)
        text_features = model.encode_text(text_tokens)
        
        # Normalize image and text features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity between image and text features
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get the predicted label indices
        _, predicted_label_index = text_probs.max(dim=1)

        # Ensure indices are within the label range
        predicted_label_index = predicted_label_index[predicted_label_index < len(labels)]
        
        return [id_label_dict[label_id.item()] for label_id in predicted_label_index]

# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Set batch size
batch_size = 300
start = 0
end = batch_size

# Iterate through the dataset in batches
while start < len(dataset):
    sample = dataset[start:end]
    img_list_batch = sample['img']
    label_id_list_batch = sample['fine_label']
    
    # Collect true labels
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list_batch])
    
    # Get predicted labels
    y_pred.extend(get_image_predict_label(images=img_list_batch))
    
    # Update batch indices
    start = end
    end += batch_size
    print(f"Processing batch: {start} - {end}")

# Print the classification report
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
