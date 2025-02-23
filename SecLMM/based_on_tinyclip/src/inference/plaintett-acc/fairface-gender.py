from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel
from datasets import load_dataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import time

# Load dataset
cifar_10_test = load_dataset('fairface', split='validation')
labels = cifar_10_test.features['gender'].names
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt
prompt = ['a photo of a male', 'a photo of a female']

# Load model and processor
model = CLIPModel.from_pretrained("clip")
processor = AutoProcessor.from_pretrained("clip")

# Ensure model is in evaluation mode
model.eval()

# Define the image prediction function
def get_image_predict_label(images):
    inputs = processor(text=prompt, images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = torch.nn.functional.softmax(logits_per_image, dim=1)
    label_ids = torch.argmax(probs, dim=1).tolist()
    return [id_label_dict[label_id] for label_id in label_ids]

# Initialize lists for true and predicted labels
y_true = []
y_pred = []

# Define batch size
batch_size = 300
start = 0
end = batch_size

# Iterate through dataset in batches
while start < len(cifar_10_test):
    sample = cifar_10_test[start:end]
    img_list, label_id_list = sample['image'], sample['gender']
    
    # Collect true labels
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list])
    
    # Get predictions
    y_pred.extend(get_image_predict_label(images=img_list))
    
    # Update batch indices
    start = end
    end += batch_size
    print(start, end)

# Print classification report
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
