from PIL import Image
import requests
from transformers import AutoProcessor, FlaxCLIPModel
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
from io import BytesIO

from datasets import load_dataset, ClassLabel

# Load dataset
cifar_10 = load_dataset('parquet', data_files={
    'train': '/home/guoyu/zw/MQBench-main/out2/train-00000-of-00001-18bc3231d015f1e8.parquet',
    'validation': '/home/guoyu/zw/MQBench-main/out2/validation-00000-of-00001-a2c3e2fba5f57a20.parquet'
})

unique_labels = sorted(set(cifar_10['train']['label']))

# Convert the label field to ClassLabel type
cifar_10 = cifar_10.cast_column('label', ClassLabel(names=unique_labels))

# Get the list of class names
labels = cifar_10['train'].features['label'].names

label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt
prompt = [f"a photo of a {label}" for label in labels]

# Load model and processor
model = FlaxCLIPModel.from_pretrained("cc")  # Replace with your model path
processor = AutoProcessor.from_pretrained("cc")  # Replace with your model path

def f_train(cifar_10, sh=5, batch_size=32):
    f_samples = []
    for label in labels:
        samples = cifar_10['train'].filter(lambda x: x['label'] == label_id_dict[label]).select(range(sh))
        f_samples.extend(samples)

    # Extract features and labels
    images = []
    sample_labels = []
    for sample in f_samples:
        try:
            # Handle binary image data
            if 'image' in sample and 'bytes' in sample['image']:
                img_bytes = sample['image']['bytes']
                img = Image.open(BytesIO(img_bytes))
                images.append(img)
                sample_labels.append(sample['label'])
            else:
                raise KeyError("Image data not found or in unsupported format.")
        except Exception as e:
            print(f"Error processing image for label {sample['label']}: {e}")
            continue

    # Check if images list is not empty
    if not images:
        raise ValueError("No valid images found in the dataset.")

    # Process images in batches
    image_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        inputs = processor(images=batch_images, return_tensors="jax", padding=True)
        batch_features = model.get_image_features(**inputs)
        image_features.append(np.array(batch_features))
    image_features = np.concatenate(image_features, axis=0)

    # Get text features
    text_inputs = processor(text=prompt, return_tensors="jax", padding=True)
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
        param_grid, cv=3, scoring='accuracy'
    )
    clf.fit(combined_features_scaled, sample_labels)

    # Print the best C value and corresponding accuracy
    print(f"Best C value: {clf.best_params_['C']}")
    print(f"Best cross-validation accuracy: {clf.best_score_:.4f}")

    return clf.best_estimator_, scaler  # Return the best model and scaler

def f_eval(clf, scaler, cifar_10_test, batch_size=32):
    # Extract test images and labels
    images = []
    test_labels = []
    for sample in cifar_10_test:
        try:
            # Handle binary image data
            if 'image' in sample and 'bytes' in sample['image']:
                img_bytes = sample['image']['bytes']
                img = Image.open(BytesIO(img_bytes))
                images.append(img)
                test_labels.append(sample['label'])
            else:
                raise KeyError("Image data not found or in unsupported format.")
        except Exception as e:
            print(f"Error processing image for label {sample['label']}: {e}")
            continue

    # Check if images list is not empty
    if not images:
        raise ValueError("No valid images found in the test dataset.")

    test_labels = jnp.array(test_labels)  # Use JAX array

    # Initialize predicted labels list
    pred_labels = []

    # Compute text features (compute once)
    text_inputs = processor(text=prompt, return_tensors="jax", padding=True)
    text_features = model.get_text_features(**text_inputs)  # Compute directly in JAX

    # Process images in batches
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting labels"):
        batch_images = images[i:i + batch_size]

        # Process images
        image_inputs = processor(images=batch_images, return_tensors="jax", padding=True)

        # Get image features
        batch_features = model.get_image_features(**image_inputs)  # Compute directly in JAX

        # Compute similarity between images and text
        combined_features = jnp.dot(batch_features, text_features.T)  # JAX dot product

        # Scale features
        combined_features_scaled = scaler.transform(combined_features)

        # Predict using the trained classifier
        batch_pred_labels = clf.predict(combined_features_scaled)  # Convert to JAX array and predict
        pred_labels.extend(batch_pred_labels)

    # Compute accuracy
    acc = accuracy_score(test_labels, pred_labels)
    return acc, pred_labels

# Run training and evaluation
sh = 10
clf, scaler = f_train(cifar_10, sh=sh)

# Save the trained linear classifier and scaler
model_filename = "f_clf_model_base_tiny.pkl"
joblib.dump(clf, model_filename)
scaler_filename = "f_scaler_base_tiny.pkl" 
joblib.dump(scaler, scaler_filename)
print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# Load the saved model and scaler
loaded_clf = joblib.load(model_filename)
loaded_scaler = joblib.load(scaler_filename)

# Evaluate the loaded model
acc, pred_labels = f_eval(loaded_clf, loaded_scaler, cifar_10['validation'])

# Print results
print(f"F accuracy (sh={sh}): {acc:.4f}")
print(classification_report(cifar_10['validation']['label'], pred_labels, target_names=labels, digits=4))
