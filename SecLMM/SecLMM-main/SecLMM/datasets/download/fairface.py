import os
import requests

# 确保 fairface 文件夹存在
os.makedirs("fairface", exist_ok=True)

# FairFace 数据集文件 URL
urls = [
    "https://huggingface.co/datasets/piadev/fairface/resolve/main/data/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/piadev/fairface/resolve/main/data/train-00001-of-00002.parquet",
    "https://huggingface.co/datasets/piadev/fairface/resolve/main/data/validation-00000-of-00001.parquet"
]

# 遍历 URL 下载文件
for url in urls:
    filename = url.split("/")[-1]  # 获取文件名
    response = requests.get(url, stream=True)  # 使用流式下载
    if response.status_code == 200:
        with open(f"fairface/{filename}", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download successful: fairface/{filename}")
    else:
        print(f"❌ Download failed: {url} (Status code: {response.status_code})")
