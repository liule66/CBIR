import os
import numpy as np
import faiss
from PIL import Image

# 导入各特征提取方法
from color import Color
from daisy import Daisy
from HOG import HOG
from resnet import extract_resnet_feature
from vggnet import extract_vgg_feature
from fusion import extract_fusion_feature  # 假设你有融合特征函数

# 使用绝对路径，避免相对路径问题
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "..", "dataset")
faiss_index_dir = os.path.join(BASE_DIR, "..", "faiss_index")
os.makedirs(faiss_index_dir, exist_ok=True)

# 特征类型与提取函数映射
feature_methods = {
    "color": lambda img: Color().histogram(img),
    "texture": lambda img: Daisy().histogram(img),
    "shape": lambda img: HOG().histogram(img),
    "resnet": extract_resnet_feature,
    "vgg": extract_vgg_feature,
    "fusion": extract_fusion_feature,  # 你可以自定义
}

for feature_type, extract_func in feature_methods.items():
    features = []
    img_paths = []
    print(f"正在提取 {feature_type} 特征...")
    for fname in os.listdir(dataset_dir):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
            img_path = os.path.join(dataset_dir, fname)
            img = Image.open(img_path)
            if img.format == 'GIF':
                try:
                    img.seek(0)
                except Exception:
                    continue
            img = img.convert('RGB')
            # 关键：根据特征类型传递不同类型
            if feature_type in ["color", "texture", "shape"]:
                input_data = np.array(img)
            else:
                input_data = img
            try:
                feat = extract_func(input_data)
                if feat is not None:
                    features.append(np.array(feat).flatten())
                    img_paths.append(fname)
            except Exception as e:
                print(f"特征提取失败: {fname}, 错误: {e}")
    if not features:
        print(f"{feature_type} 没有提取到任何特征，跳过。")
        continue
    features = np.vstack(features).astype('float32')
    np.save(os.path.join(faiss_index_dir, f'features_{feature_type}.npy'), features)
    with open(os.path.join(faiss_index_dir, f'img_paths_{feature_type}.txt'), 'w', encoding='utf-8') as f:
        for p in img_paths:
            f.write(p+'\n')
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    faiss.write_index(index, os.path.join(faiss_index_dir, f'index_{feature_type}.faiss'))
    print(f"{feature_type} 特征索引构建完成，共处理了 {len(features)} 张图片。") 