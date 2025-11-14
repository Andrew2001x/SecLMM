import os
import requests
os.makedirs("tiny-imagenet", exist_ok=True)
urls = [
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet",
    "https://huggingface.co/datasets/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet"
]
for url in urls:
    filename = url.split("/")[-1]  
    response = requests.get(url, stream=True)  
    if response.status_code == 200:
        with open(f"tiny-imagenet/{filename}", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download successfully: tiny-imagenet/{filename}")
    else:
        print(f"❌ Download failed: {url} (Status code: {response.status_code})")
