import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression




from datasets import load_dataset
import numpy as np
from tqdm import tqdm
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
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
cifar_100 = load_dataset('cifar100')
labels = cifar_100['train'].features['fine_label'].names
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt
prompt = [f"a photo of a {label}" for label in labels]

# Load model and processor
model = FlaxCLIPModel.from_pretrained("cc")  # Replace with your model path
processor = AutoProcessor.from_pretrained("cc")  # Replace with your model path

def f_train(cifar_100, sh=5):
    f_samples = []
    for label in labels:
        samples = cifar_100['train'].filter(lambda x: x['fine_label'] == label_id_dict[label]).select(range(sh))
        f_samples.extend(samples)
    
    # Extract features and labels
    images = [sample['img'] for sample in f_samples]
    sample_labels = [sample['fine_label'] for sample in f_samples]
    
    # Get image features
    inputs = processor(images=images, return_tensors="jax", padding=True)
    with jax.disable_jit():
        image_features = np.array(model.get_image_features(**inputs))
    
    # Get text features
    text_inputs = processor(text=prompt, return_tensors="jax", padding=True)
    with jax.disable_jit():
        text_features = np.array(model.get_text_features(**text_inputs))
    
    # Combine image and text features
    combined_features = image_features @ text_features.T
    
    # Scale features
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    # Use GridSearchCV to find the best C value
    param_grid = {'C': np.arange(0.1, 10.1, 0.1)}  # C from 0.1 to 10, step 0.1
    clf = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=0, solver='liblinear'),
        param_grid,
        cv=3,
        scoring='accuracy'
    )
    clf.fit(combined_features_scaled, sample_labels)
    
    # Print the best C value and corresponding accuracy
    print(f"Best C value: {clf.best_params_['C']}")
    print(f"Best cross-validation accuracy: {clf.best_score_:.4f}")
    
    # Print all C values and their corresponding accuracies
    for params, mean_score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score']):
        print(f"C={params['C']:.1f}, Accuracy={mean_score:.4f}")
    
    return clf.best_estimator_, scaler  # Return the best model and scaler


sh = 10
clf, scaler = f_train(cifar_100, sh=sh)

# Save the trained linear classifier and scaler
model_filename = "f_clf_model_base_cifar100.pkl"
joblib.dump(clf, model_filename)
scaler_filename = "f_scaler_base_cifar100.pkl"
joblib.dump(scaler, scaler_filename)
print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")


# Load saved classifier and scaler
model_filename = "f_clf_model_base_cifar100.pkl"
scaler_filename = "f_scaler_base_cifar100.pkl"

clf = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Function to load features from text files
def load_features(filename):
    with open(filename, 'r') as f:
        features = [list(map(float, line.strip().split())) for line in f]
    return np.array(features)

# Load image and text features
image_features = load_features("outputs_image_cifar100.json")
text_features = load_features("outputs_text_cifar100.json")

# Combine image and text features (dot product)
combined_features = np.dot(image_features, text_features.T)

# Scale features
combined_features_scaled = scaler.transform(combined_features)

# Predict labels
pred_labels = clf.predict(combined_features_scaled)

# Print predictions
print("Predicted Labels:", pred_labels)
