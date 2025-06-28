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
from fusion import extract_fusion_feature  

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
    "fusion": extract_fusion_feature,
}

def get_image_files(directory):
    """获取目录中的所有图片文件"""
    image_files = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
            image_files.append(fname)
    return image_files

def extract_features_from_image(img_path, feature_type, extract_func):
    """从单张图片提取特征"""
    try:
        img = Image.open(img_path)
        if img.format == 'GIF':
            try:
                img.seek(0)
            except Exception:
                return None
        img = img.convert('RGB')
        
        # 根据特征类型传递不同类型
        if feature_type in ["color", "texture", "shape"]:
            input_data = np.array(img)
        else:
            input_data = img
            
        feat = extract_func(input_data)
        if feat is not None:
            return np.array(feat).flatten()
    except Exception as e:
        print(f"特征提取失败: {img_path}, 错误: {e}")
    return None

def main():
    # 获取所有图片文件
    image_files = get_image_files(dataset_dir)
    if not image_files:
        print("dataset目录中没有找到图片文件。")
        return
    
    print(f"发现 {len(image_files)} 张图片，开始完整索引构建...")
    
    # 为每种特征类型构建完整索引
    for feature_type, extract_func in feature_methods.items():
        features = []
        img_paths = []
        print(f"正在提取 {feature_type} 特征...")
        
        for fname in image_files:
            img_path = os.path.join(dataset_dir, fname)
            feat = extract_features_from_image(img_path, feature_type, extract_func)
            if feat is not None:
                features.append(feat)
                img_paths.append(fname)
        
        if not features:
            print(f"{feature_type} 没有提取到任何特征，跳过。")
            continue
        
        # 保存特征和路径
        features = np.vstack(features).astype('float32')
        features_file = os.path.join(faiss_index_dir, f'features_{feature_type}.npy')
        paths_file = os.path.join(faiss_index_dir, f'img_paths_{feature_type}.txt')
        index_file = os.path.join(faiss_index_dir, f'index_{feature_type}.faiss')
        
        np.save(features_file, features)
        with open(paths_file, 'w', encoding='utf-8') as f:
            for p in img_paths:
                f.write(p + '\n')
        
        # 创建FAISS索引
        index = faiss.IndexFlatL2(features.shape[1])
        index.add(features)
        faiss.write_index(index, index_file)
        
        print(f"{feature_type} 特征索引构建完成，共处理了 {len(features)} 张图片。")
    
    print("完整索引构建完成！")

if __name__ == "__main__":
    main() 