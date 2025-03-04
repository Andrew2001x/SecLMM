import os
import requests
os.makedirs("cifar100", exist_ok=True)
urls = [
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet",
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/train-00000-of-00001.parquet"
]
for url in urls:
    filename = url.split("/")[-1]  
    response = requests.get(url, stream=True)  
    if response.status_code == 200:
        with open(f"cifar100/{filename}", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download successfully: cifar100/{filename}")
    else:
        print(f"❌ Download failed: {url} (Status code: {response.status_code})")
