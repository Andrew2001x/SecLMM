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

# 加载模型和变换
arch = 'TinyCLIP-ViT-39M-16-Text-19M'
checkpoint_path = '/jty/zhangwang/Azhangwang/MQBench/dist/mo/yuanshi.pt'

model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='/jty/zhangwang/Azhangwang/MQBench/dist/mo/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt')
##model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained='YFCC15M')

# 确保模型处于推理模式
model.eval()
dataset = load_dataset('out1', split='test')  # 这里加载 CIFAR-10 测试集
#dataset = dataset.select(range(600))  
# 设置文本描述
tokenizer = open_clip.get_tokenizer(arch)
labels = dataset.features['fine_label'].names
prompt = ['a photo of a apple', 'a photo of a aquarium_fish', 'a photo of a baby', 'a photo of a bear', 'a photo of a beaver', 'a photo of a bed', 'a photo of a bee', 'a photo of a beetle', 'a photo of a bicycle', 'a photo of a bottle', 'a photo of a bowl', 'a photo of a boy', 'a photo of a bridge', 'a photo of a bus', 'a photo of a butterfly', 'a photo of a camel', 'a photo of a can', 'a photo of a castle', 'a photo of a caterpillar', 'a photo of a cattle', 'a photo of a chair', 'a photo of a chimpanzee', 'a photo of a clock', 'a photo of a cloud', 'a photo of a cockroach', 'a photo of a couch', 'a photo of a cra', 'a photo of a crocodile', 'a photo of a cup', 'a photo of a dinosaur', 'a photo of a dolphin', 'a photo of a elephant', 'a photo of a flatfish', 'a photo of a forest', 'a photo of a fox', 'a photo of a girl', 'a photo of a hamster', 'a photo of a house', 'a photo of a kangaroo', 'a photo of a keyboard', 'a photo of a lamp', 'a photo of a lawn_mower', 'a photo of a leopard', 'a photo of a lion', 'a photo of a lizard', 'a photo of a lobster', 'a photo of a man', 'a photo of a maple_tree', 'a photo of a motorcycle', 'a photo of a mountain', 'a photo of a mouse', 'a photo of a mushroom', 'a photo of a oak_tree', 'a photo of a orange', 'a photo of a orchid', 'a photo of a otter', 'a photo of a palm_tree', 'a photo of a pear', 'a photo of a pickup_truck', 'a photo of a pine_tree', 'a photo of a plain', 'a photo of a plate', 'a photo of a poppy', 'a photo of a porcupine', 'a photo of a possum', 'a photo of a rabbit', 'a photo of a raccoon', 'a photo of a ray', 'a photo of a road', 'a photo of a rocket', 'a photo of a rose', 'a photo of a sea', 'a photo of a seal', 'a photo of a shark', 'a photo of a shrew', 'a photo of a skunk', 'a photo of a skyscraper', 'a photo of a snail', 'a photo of a snake', 'a photo of a spider', 'a photo of a squirrel', 'a photo of a streetcar', 'a photo of a sunflower', 'a photo of a sweet_pepper', 'a photo of a table', 'a photo of a tank', 'a photo of a telephone', 'a photo of a television', 'a photo of a tiger', 'a photo of a tractor', 'a photo of a train', 'a photo of a trout', 'a photo of a tulip', 'a photo of a turtle', 'a photo of a wardrobe', 'a photo of a whale', 'a photo of a willow_tree', 'a photo of a wolf', 'a photo of a woman', 'a photo of a worm']


# 创建标签字典
label_id_dict = {label: idx for idx, label in enumerate(labels)}
id_label_dict = {idx: label for idx, label in enumerate(labels)}

# 加载数据集

img_list = dataset['img']  # 图像字段
label_id_list = dataset['fine_label']  # 标签字段

# 定义批量预测函数
def get_image_predict_label(images):
    # 将图片通过预处理函数转换
    inputs = [preprocess(image) for image in images]
    inputs = torch.stack(inputs)
    
    # 执行推理
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        
        # 将文本描述转换为 token
        text_tokens = tokenizer(prompt)
        text_features = model.encode_text(text_tokens)
        
        # 对图像和文本特征进行归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 计算图像和文本的相似度
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # 获取预测标签的索引
        _, predicted_label_index = text_probs.max(dim=1)

        # 确保索引在标签范围内
        predicted_label_index = predicted_label_index[predicted_label_index < len(labels)]
        
        return [id_label_dict[label_id.item()] for label_id in predicted_label_index]

# 初始化实际标签和预测标签的列表
y_true = []
y_pred = []

# 设置批处理大小
batch_size = 300
start = 0
end = batch_size

# 遍历数据集并进行批量推理
while start < len(dataset):
    sample = dataset[start:end]
    img_list_batch = sample['img']
    label_id_list_batch = sample['fine_label']
    
    # 收集实际标签
    y_true.extend([id_label_dict[label_id] for label_id in label_id_list_batch])
    
    # 获取预测标签
    y_pred.extend(get_image_predict_label(images=img_list_batch))
    
    # 更新批次索引
    start = end
    end += batch_size
    print(f"Processing batch: {start} - {end}")

# 打印分类报告
print(classification_report(y_true, y_pred, target_names=labels, digits=4))
