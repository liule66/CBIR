import os
import numpy as np
import faiss
from PIL import Image
import shutil
import json
import torch
import gc

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
new_dir = os.path.join(dataset_dir, "new")
faiss_index_dir = os.path.join(BASE_DIR, "..", "faiss_index")
os.makedirs(faiss_index_dir, exist_ok=True)
os.makedirs(new_dir, exist_ok=True)

# 特征类型与提取函数映射
feature_methods = {
    "color": lambda img: Color().histogram(img),
    "texture": lambda img: Daisy().histogram(img),
    "shape": lambda img: HOG().histogram(img),
    "resnet": extract_resnet_feature,
    "vgg": extract_vgg_feature,
    "fusion": extract_fusion_feature,
}

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

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
        # 清理GPU内存
        clear_gpu_memory()
    return None

def load_existing_index(feature_type):
    """加载现有的索引文件"""
    features_file = os.path.join(faiss_index_dir, f'features_{feature_type}.npy')
    paths_file = os.path.join(faiss_index_dir, f'img_paths_{feature_type}.txt')
    index_file = os.path.join(faiss_index_dir, f'index_{feature_type}.faiss')
    
    features = []
    img_paths = []
    index = None
    
    if os.path.exists(features_file) and os.path.exists(paths_file) and os.path.exists(index_file):
        try:
            features = np.load(features_file)
            with open(paths_file, 'r', encoding='utf-8') as f:
                img_paths = [line.strip() for line in f]
            index = faiss.read_index(index_file)
            print(f"加载现有 {feature_type} 索引，包含 {len(features)} 张图片")
        except Exception as e:
            print(f"加载现有索引失败: {e}")
            features = []
            img_paths = []
            index = None
    
    return features, img_paths, index

def save_index(features, img_paths, index, feature_type):
    """保存索引文件"""
    features_file = os.path.join(faiss_index_dir, f'features_{feature_type}.npy')
    paths_file = os.path.join(faiss_index_dir, f'img_paths_{feature_type}.txt')
    index_file = os.path.join(faiss_index_dir, f'index_{feature_type}.faiss')
    
    np.save(features_file, features)
    with open(paths_file, 'w', encoding='utf-8') as f:
        for p in img_paths:
            f.write(p + '\n')
    faiss.write_index(index, index_file)

def move_new_images_to_dataset():
    """将new目录中的图片移动到dataset目录"""
    new_images = get_image_files(new_dir)
    moved_files = {}
    
    for fname in new_images:
        src_path = os.path.join(new_dir, fname)
        
        # 检查目标文件是否已存在，如果存在则重命名
        base_name, ext = os.path.splitext(fname)
        counter = 1
        new_fname = fname
        dst_path = os.path.join(dataset_dir, new_fname)
        
        while os.path.exists(dst_path):
            new_fname = f"{base_name}_{counter}{ext}"
            dst_path = os.path.join(dataset_dir, new_fname)
            counter += 1
        
        # 移动文件
        shutil.move(src_path, dst_path)
        moved_files[fname] = new_fname
        print(f"移动文件: {fname} -> {new_fname}")
    
    return moved_files

def process_feature_type(feature_type, extract_func, moved_files):
    """处理单个特征类型，包含内存管理"""
    print(f"正在处理 {feature_type} 特征...")
    
    # 加载现有索引
    existing_features, existing_paths, existing_index = load_existing_index(feature_type)
    
    # 提取新图片的特征
    new_features = []
    new_paths = []
    
    for old_fname, new_fname in moved_files.items():
        img_path = os.path.join(dataset_dir, new_fname)
        feat = extract_features_from_image(img_path, feature_type, extract_func)
        if feat is not None:
            new_features.append(feat)
            new_paths.append(new_fname)
        
        # 每处理几张图片就清理一次内存
        if len(new_features) % 3 == 0:
            clear_gpu_memory()
    
    if not new_features:
        print(f"{feature_type} 没有提取到任何新特征，跳过。")
        return
    
    # 检查特征维度一致性
    feature_dims = [feat.shape[0] for feat in new_features]
    if len(set(feature_dims)) > 1:
        print(f"警告：{feature_type} 特征维度不一致: {feature_dims}")
        # 使用最小维度，截断其他特征
        min_dim = min(feature_dims)
        new_features = [feat[:min_dim] for feat in new_features]
        print(f"统一特征维度为: {min_dim}")
    
    # 合并特征和路径
    new_features = np.vstack(new_features).astype('float32')
    if len(existing_features) > 0:
        # 检查维度匹配
        if existing_features.shape[1] != new_features.shape[1]:
            print(f"警告：现有特征维度 {existing_features.shape[1]} 与新特征维度 {new_features.shape[1]} 不匹配")
            # 使用较小的维度
            min_dim = min(existing_features.shape[1], new_features.shape[1])
            existing_features = existing_features[:, :min_dim]
            new_features = new_features[:, :min_dim]
        
        all_features = np.vstack([existing_features, new_features])
        all_paths = existing_paths + new_paths
    else:
        all_features = new_features
        all_paths = new_paths
    
    # 创建或更新FAISS索引
    if existing_index is not None:
        # 更新现有索引
        existing_index.add(new_features)
        index = existing_index
    else:
        # 创建新索引
        index = faiss.IndexFlatL2(all_features.shape[1])
        index.add(all_features)
    
    # 保存更新后的索引
    save_index(all_features, all_paths, index, feature_type)
    print(f"{feature_type} 特征索引更新完成，新增 {len(new_features)} 张图片，总计 {len(all_features)} 张图片。")
    
    # 清理内存
    clear_gpu_memory()

def main():
    # 检查是否有新图片需要处理
    new_images = get_image_files(new_dir)
    if not new_images:
        print("没有发现新图片，跳过增量索引构建。")
        print("请将新图片放入 dataset/new/ 目录中。")
        return
    
    print(f"发现 {len(new_images)} 张新图片，开始增量索引构建...")
    
    # 移动新图片到dataset目录
    moved_files = move_new_images_to_dataset()
    
    # 为每种特征类型构建增量索引
    for feature_type, extract_func in feature_methods.items():
        try:
            process_feature_type(feature_type, extract_func, moved_files)
        except Exception as e:
            print(f"处理 {feature_type} 特征时出错: {e}")
            clear_gpu_memory()
            continue
    
    # 保存文件重命名映射（可选，用于调试）
    mapping_file = os.path.join(faiss_index_dir, 'file_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(moved_files, f, ensure_ascii=False, indent=2)
    
    print("增量索引构建完成！")

if __name__ == "__main__":
    main() 