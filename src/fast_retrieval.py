import numpy as np
import faiss
import pickle
import hashlib
from collections import OrderedDict
import time
from typing import List, Tuple, Dict, Any
import os

class LRUCache:
    """LRU缓存实现"""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Any:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class FastRetrieval:
    """高性能图像检索系统"""
    
    def __init__(self, 
                 index_dir: str,
                 cache_size: int = 1000,
                 use_hierarchical: bool = True,
                 filter_ratio: float = 0.1):
        """
        初始化快速检索系统
        
        Args:
            index_dir: 索引文件目录
            cache_size: 缓存大小
            use_hierarchical: 是否使用分层检索
            filter_ratio: 预过滤比例
        """
        self.index_dir = index_dir
        self.cache = LRUCache(cache_size)
        self.use_hierarchical = use_hierarchical
        self.filter_ratio = filter_ratio
        
        # 加载索引
        self.indices = {}
        self.features = {}
        self.img_paths = {}
        self.load_indices()
        
        # 构建轻量级索引用于预过滤
        if self.use_hierarchical:
            self.build_lightweight_index()
    
    def load_indices(self):
        """加载所有特征索引"""
        feature_types = ['color', 'texture', 'shape', 'resnet', 'vgg', 'fusion']
        
        for feature_type in feature_types:
            try:
                # 加载特征
                features_file = os.path.join(self.index_dir, f'features_{feature_type}.npy')
                if os.path.exists(features_file):
                    self.features[feature_type] = np.load(features_file)
                
                # 加载索引
                index_file = os.path.join(self.index_dir, f'index_{feature_type}.faiss')
                if os.path.exists(index_file):
                    self.indices[feature_type] = faiss.read_index(index_file)
                
                # 加载图片路径
                paths_file = os.path.join(self.index_dir, f'img_paths_{feature_type}.txt')
                if os.path.exists(paths_file):
                    with open(paths_file, 'r', encoding='utf-8') as f:
                        self.img_paths[feature_type] = [line.strip() for line in f]
                
                print(f"加载 {feature_type} 索引，包含 {len(self.features.get(feature_type, []))} 张图片")
                
            except Exception as e:
                print(f"加载 {feature_type} 索引失败: {e}")
    
    def build_lightweight_index(self):
        """构建轻量级索引用于预过滤"""
        if 'color' in self.features:
            # 使用颜色特征作为轻量级索引
            color_features = self.features['color']
            self.lightweight_index = faiss.IndexFlatL2(color_features.shape[1])
            self.lightweight_index.add(color_features.astype('float32'))
            print(f"构建轻量级索引，维度: {color_features.shape[1]}")
    
    def compute_query_hash(self, query_feature: np.ndarray) -> str:
        """计算查询特征哈希值用于缓存"""
        return hashlib.md5(query_feature.tobytes()).hexdigest()
    
    def hierarchical_search(self, 
                          query_feature: np.ndarray, 
                          feature_type: str, 
                          k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        分层检索：先用轻量级特征过滤，再用精确特征检索
        
        Args:
            query_feature: 查询特征
            feature_type: 特征类型
            k: 返回结果数量
            
        Returns:
            distances: 距离数组
            indices: 索引数组
            img_paths: 图片路径列表
        """
        if not self.use_hierarchical or 'color' not in self.features:
            # 直接检索
            return self.direct_search(query_feature, feature_type, k)
        
        # 第一步：使用颜色特征快速过滤
        color_feature = query_feature[:self.features['color'].shape[1]]  # 假设颜色特征在前
        filter_k = max(k * 10, int(len(self.features['color']) * self.filter_ratio))
        
        D_filter, I_filter = self.lightweight_index.search(
            color_feature.reshape(1, -1).astype('float32'), 
            filter_k
        )
        
        # 第二步：在候选集中使用精确特征检索
        if feature_type in self.features:
            candidate_features = self.features[feature_type][I_filter[0]]
            candidate_paths = [self.img_paths[feature_type][i] for i in I_filter[0]]
            
            # 计算精确相似度
            distances = np.linalg.norm(candidate_features - query_feature, axis=1)
            top_indices = np.argsort(distances)[:k]
            
            return (distances[top_indices], 
                   I_filter[0][top_indices], 
                   [candidate_paths[i] for i in top_indices])
        
        return np.array([]), np.array([]), []
    
    def direct_search(self, 
                     query_feature: np.ndarray, 
                     feature_type: str, 
                     k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """直接检索"""
        if feature_type not in self.indices:
            return np.array([]), np.array([]), []
        
        D, I = self.indices[feature_type].search(
            query_feature.reshape(1, -1).astype('float32'), 
            k
        )
        
        img_paths = [self.img_paths[feature_type][i] for i in I[0]]
        return D[0], I[0], img_paths
    
    def search(self, 
              query_feature: np.ndarray, 
              feature_type: str = 'fusion', 
              k: int = 5,
              use_cache: bool = True) -> Dict[str, Any]:
        """
        执行检索
        
        Args:
            query_feature: 查询特征
            feature_type: 特征类型
            k: 返回结果数量
            use_cache: 是否使用缓存
            
        Returns:
            检索结果字典
        """
        start_time = time.time()
        
        # 计算查询哈希
        query_hash = self.compute_query_hash(query_feature)
        
        # 检查缓存
        if use_cache:
            cached_result = self.cache.get(query_hash)
            if cached_result is not None:
                cached_result['cache_hit'] = True
                cached_result['search_time'] = time.time() - start_time
                return cached_result
        
        # 执行检索
        if self.use_hierarchical:
            distances, indices, img_paths = self.hierarchical_search(
                query_feature, feature_type, k
            )
        else:
            distances, indices, img_paths = self.direct_search(
                query_feature, feature_type, k
            )
        
        # 计算相似度分数
        if len(distances) > 0:
            d_min, d_max = float(np.min(distances)), float(np.max(distances))
            if d_max > d_min:
                scores = [1 - (float(d) - d_min) / (d_max - d_min) for d in distances]
            else:
                scores = [1.0 for _ in distances]
        else:
            scores = []
        
        # 构建结果
        result = {
            'feature_type': feature_type,
            'distances': distances.tolist() if len(distances) > 0 else [],
            'indices': indices.tolist() if len(indices) > 0 else [],
            'img_paths': img_paths,
            'scores': scores,
            'search_time': time.time() - start_time,
            'cache_hit': False,
            'total_images': len(self.features.get(feature_type, []))
        }
        
        # 缓存结果
        if use_cache:
            self.cache.put(query_hash, result)
        
        return result
    
    def batch_search(self, 
                    query_features: List[np.ndarray], 
                    feature_type: str = 'fusion', 
                    k: int = 5) -> List[Dict[str, Any]]:
        """批量检索"""
        results = []
        for query_feature in query_features:
            result = self.search(query_feature, feature_type, k)
            results.append(result)
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        stats = {}
        for feature_type, features in self.features.items():
            stats[feature_type] = {
                'num_images': len(features),
                'feature_dim': features.shape[1] if len(features) > 0 else 0,
                'index_type': type(self.indices.get(feature_type)).__name__
            }
        return stats
    
    def optimize_index(self, feature_type: str):
        """优化索引性能"""
        if feature_type in self.indices:
            index = self.indices[feature_type]
            
            # 如果是IVF索引，训练量化器
            if hasattr(index, 'train'):
                features = self.features[feature_type]
                index.train(features.astype('float32'))
            
            # 重建索引
            if hasattr(index, 'make_direct_map'):
                index.make_direct_map()
            
            print(f"优化 {feature_type} 索引完成")

# 使用示例
if __name__ == "__main__":
    # 初始化快速检索系统
    fast_retrieval = FastRetrieval(
        index_dir="../faiss_index",
        cache_size=1000,
        use_hierarchical=True,
        filter_ratio=0.1
    )
    
    # 获取索引统计
    stats = fast_retrieval.get_index_stats()
    print("索引统计:", stats)
    
    # 测试检索性能
    if 'fusion' in fast_retrieval.features:
        test_feature = fast_retrieval.features['fusion'][0]  # 使用第一张图片作为查询
        
        # 测试直接检索
        result_direct = fast_retrieval.search(
            test_feature, 'fusion', k=5, use_cache=False
        )
        print(f"直接检索时间: {result_direct['search_time']:.4f}秒")
        
        # 测试分层检索
        result_hierarchical = fast_retrieval.search(
            test_feature, 'fusion', k=5, use_cache=True
        )
        print(f"分层检索时间: {result_hierarchical['search_time']:.4f}秒")
        print(f"缓存命中: {result_hierarchical['cache_hit']}") 