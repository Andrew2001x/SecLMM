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

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
np.random.seed(seed)
random.seed(seed)

# For deterministic algorithms in CuDNN (for GPUs)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和变换
arch = 'TinyCLIP-ViT-39M-16-Text-19M'
checkpoint_path = '/na/mq/dist/mo/yuanshi.pt'

# 创建模型并转移到 GPU
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='/na/mq/dist/mo/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt')
model.to(device)  # 将模型移动到GPU

# 确保模型处于推理模式
model.eval()
dataset = load_dataset('out3', split='validation')  # 这里加载 CIFAR-10 测试集
#dataset = dataset.select(range(600))  
labels = dataset.features['age'].names
prompt = ['a photo of people between the ages of 0 and 2','a photo of people between the ages of 3 and 9', 'a photo of people between the ages of 10 and 19', 'a photo of people between the ages of 20 and 29', 'a photo of people between the ages of 30 and 39', 'a photo of people between the ages of 40 and 49', 'a photo of people between the ages of 50 and 59',  'a photo of people between the ages of 60 and 69','a photo of people aged 70 or older' ]

# 设置文本描述
tokenizer = open_clip.get_tokenizer(arch)

# 创建标签字典
label_id_dict = {label: idx for idx, label in enumerate(labels)}
id_label_dict = {idx: label for idx, label in enumerate(labels)}

# 加载数据集
img_list = dataset['image']  # 图像字段
label_id_list = dataset['age']  # 标签字段

# 定义批量预测函数
def get_image_predict_label(images):
    # 将图片通过预处理函数转换
    inputs = [preprocess(image) for image in images]
    inputs = torch.stack(inputs).to(device)  # 将数据移动到GPU
    
    # 执行推理
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        
        # 将文本描述转换为 token
        text_tokens = tokenizer(prompt).to(device)  # 将文本输入移动到GPU
        text_features = model.encode_text(text_tokens)
        
        # 对图像和文本特征进行归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 计算图像和文本的相似度
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # 获取预测标签的索引
        _, predicted_label_index = text_probs.max(dim=1)
        return [id_label_dict[label_id.item()] for label_id in predicted_label_index]

# 初始化实际标签和预测标签的列表
y_true = []
y_pred = []

# 设置批处理大小
batch_size = 300
total_samples = len(dataset)

# 遍历数据集并进行批量推理
for start in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
    end = min(start + batch_size, total_samples)  # 确保不会超出数据集的范围
    sample = dataset[start:end]
    img_list_batch = sample['image']
    label_id_list_batch = sample['age']
    
    # 收集实际标签
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list_batch])
    
    # 获取预测标签
    y_pred.extend(get_image_predict_label(images=img_list_batch))

# 打印分类报告
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
