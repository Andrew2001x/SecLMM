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

# Load dataset
#cifar_10 = load_dataset('out3')

cifar_10 = load_dataset('parquet', data_files={
    'train': '/home/guoyu/zw/MQBench-main/out3/train-00000-of-00002-d405faba4f4b9b85.parquet',
    'validation': '/home/guoyu/zw/MQBench-main/out3/validation-00000-of-00001-951dbd63c8724ee1.parquet'
})



labels = cifar_10['train'].features['age'].names
label_id_dict = dict(zip(labels, range(len(labels))))
id_label_dict = dict(zip(range(len(labels)), labels))

# Prepare the prompt
prompt = [f"a photo of a {label}" for label in labels]

# Load model and processor
model = FlaxCLIPModel.from_pretrained("cc")  # Replace with your model path
processor = AutoProcessor.from_pretrained("cc")  # Replace with your model path

def f_train(cifar_10, sh=5):
    f_samples = []
    for label in labels:
        samples = cifar_10['train'].filter(lambda x: x['age'] == label_id_dict[label]).select(range(sh))
        f_samples.extend(samples)
    
    # Extract features and labels
    images = [sample['image'] for sample in f_samples]
    sample_labels = [sample['age'] for sample in f_samples]
    
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

def f_eval(clf, scaler, cifar_10_test, batch_size=200):
    # 提取测试图像和标签
    images = [sample['image'] for sample in cifar_10_test]
    test_labels = jnp.array([sample['age'] for sample in cifar_10_test])  # 使用 JAX 数组

    # 初始化预测标签列表
    pred_labels = []

    # 计算文本特征（只计算一次）
    text_inputs = processor(text=prompt, return_tensors="jax", padding=True)
    text_features = model.get_text_features(**text_inputs)  # 直接在 JAX 中计算

    # 批量处理图像
    for i in tqdm(range(0, len(images), batch_size), desc="Predicting labels"):
        batch_images = images[i:i + batch_size]
        
        # 处理图像
        image_inputs = processor(images=batch_images, return_tensors="jax", padding=True)
        
        # 获取图像特征
        batch_features = model.get_image_features(**image_inputs)  # 直接在 JAX 中计算
        
        # 计算图像和文本的相似度
        combined_features = jnp.dot(batch_features, text_features.T)  # JAX 计算点积
        
        # 缩放特征
        combined_features_scaled = scaler.transform(combined_features)
        
        # 使用训练好的分类器进行预测
        batch_pred_labels = clf.predict(combined_features_scaled)  # 转换为 JAX 数组后预测
        pred_labels.extend(batch_pred_labels)
    
    # 计算准确率
    acc = accuracy_score(test_labels, pred_labels)
    return acc, pred_labels

# Run training and evaluation
sh = 10
clf, scaler = f_train(cifar_10, sh=sh)

# Save the trained linear classifier and scaler
model_filename = "f_clf_model_base_fairage.pkl"
joblib.dump(clf, model_filename)
scaler_filename = "f_scaler_base_fairage.pkl"
joblib.dump(scaler, scaler_filename)
print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")


# Load saved classifier and scaler
model_filename = "f_clf_model_base_fairage.pkl"
scaler_filename = "f_scaler_base_fairage.pkl"

clf = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)

# Function to load features from text files
def load_features(filename):
    with open(filename, 'r') as f:
        features = [list(map(float, line.strip().split())) for line in f]
    return np.array(features)

# Load image and text features
image_features = load_features("outputs_image_fairage.json")
text_features = load_features("outputs_text_fairage.json")

# Combine image and text features (dot product)
combined_features = np.dot(image_features, text_features.T)

# Scale features
combined_features_scaled = scaler.transform(combined_features)

# Predict labels
pred_labels = clf.predict(combined_features_scaled)

# Print predictions
print("Predicted Labels:", pred_labels)
