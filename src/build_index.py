import os
import numpy as np
import faiss
from PIL import Image
from resnet import extract_resnet_feature  # 请确保此函数已实现

# 使用绝对路径，避免相对路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "..", "dataset")
features = []
img_paths = []

# 检查dataset目录是否存在
if not os.path.exists(dataset_dir):
    print(f"错误：找不到dataset目录: {dataset_dir}")
    print("请确保在CBIR项目根目录下有dataset文件夹")
    exit(1)

print(f"正在处理dataset目录: {dataset_dir}")

for fname in os.listdir(dataset_dir):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(dataset_dir, fname)
        print(f"处理图片: {fname}")
        img = Image.open(img_path).convert('RGB')
        feat = extract_resnet_feature(img)  # shape: (feature_dim,)
        if feat is not None:
            features.append(feat)
            img_paths.append(fname)
        else:
            print(f"警告：无法提取特征: {fname}")

if not features:
    print("错误：没有找到任何图片或无法提取特征")
    exit(1)

features = np.vstack(features).astype('float32')
faiss_index_dir = os.path.join(BASE_DIR, "..", "faiss_index")
os.makedirs(faiss_index_dir, exist_ok=True)
np.save(os.path.join(faiss_index_dir, 'features.npy'), features)
with open(os.path.join(faiss_index_dir, 'img_paths.txt'), 'w', encoding='utf-8') as f:
    for p in img_paths:
        f.write(p+'\n')

index = faiss.IndexFlatL2(features.shape[1])
index.add(features)
faiss.write_index(index, os.path.join(faiss_index_dir, 'index.faiss'))
print(f"FAISS索引构建完成！共处理了 {len(features)} 张图片") 