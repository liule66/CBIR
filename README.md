[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

# CBIR (Content-Based Image Retrieval) 系统

基于内容的图像检索系统，使用深度学习特征和FAISS向量索引实现高效的相似图像搜索。

## 项目简介

本项目实现了一个完整的基于内容的图像检索系统，具有以下特点：

- **多种特征提取方法**：支持ResNet、VGG、颜色直方图、HOG等多种特征
- **高效向量检索**：使用FAISS进行快速相似性搜索
- **Web界面**：基于FastAPI的现代化Web界面
- **模块化设计**：易于扩展和维护

## 功能特性

- 🖼️ 支持多种图像格式（JPG、PNG、JPEG）
- 🔍 基于深度学习的图像特征提取
- ⚡ 高效的向量相似性搜索
- 🌐 现代化的Web用户界面
- 📊 支持批量处理和索引构建
- 🔧 可配置的特征提取参数

## 技术栈

- **后端框架**：FastAPI
- **深度学习**：PyTorch + torchvision
- **向量检索**：FAISS
- **图像处理**：PIL (Pillow)
- **前端模板**：Jinja2

## 安装说明

### 环境要求

- Python 3.7+
- PyTorch
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/liule66/CBIR.git
cd CBIR
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据集**
```bash
# 将图片放入dataset目录
mkdir dataset
# 或者使用CIFAR-10数据集
python src/cifar2img.py
```

4. **构建索引**
```bash
python src/build_index.py
```

5. **启动服务**
```bash
uvicorn src.web_main:app --reload --host 0.0.0.0 --port 8000
```

6. **访问系统**
打开浏览器访问：http://localhost:8000

## 项目结构

```
CBIR/
├── src/                    # 源代码目录
│   ├── web_main.py        # FastAPI主应用
│   ├── build_index.py     # 索引构建脚本
│   ├── resnet.py          # ResNet特征提取
│   ├── vggnet.py          # VGG特征提取
│   ├── color.py           # 颜色特征提取
│   ├── HOG.py             # HOG特征提取
│   ├── edge.py            # 边缘特征提取
│   ├── gabor.py           # Gabor特征提取
│   ├── daisy.py           # DAISY特征提取
│   ├── fusion.py          # 特征融合
│   ├── evaluate.py        # 评估模块
│   ├── DB.py              # 数据库操作
│   ├── utils.py           # 工具函数
│   └── cifar2img.py       # CIFAR-10转换脚本
├── templates/              # HTML模板
│   ├── index.html         # 主页
│   └── result.html        # 结果页
├── dataset/               # 图片数据集
├── faiss_index/           # FAISS索引文件
├── cache/                 # 缓存文件
├── result/                # 评估结果
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
└── USAGE.md              # 使用指南
```

## 使用方法

### 1. 准备数据集

将图片文件放入 `dataset/` 目录，支持JPG、PNG、JPEG格式。

### 2. 构建索引

```bash
python src/build_index.py
```

这将：
- 遍历 `dataset/` 目录中的所有图片
- 提取ResNet特征
- 构建FAISS索引
- 保存索引文件到 `faiss_index/` 目录

### 3. 启动Web服务

```bash
uvicorn src.web_main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 使用Web界面

1. 打开浏览器访问 http://localhost:8000
2. 上传一张图片
3. 系统将返回最相似的5张图片

## 配置说明

### 特征提取配置

在 `src/resnet.py` 中可以修改：
- `RES_model`：ResNet模型版本（resnet18/34/50/101/152）
- `pick_layer`：特征层选择（avg/max/fc）
- `d_type`：距离度量类型

### 检索参数配置

在 `src/web_main.py` 中可以修改：
- 返回结果数量（默认5张）
- 距离度量方式

## 评估指标

系统支持多种评估指标：
- **MMAP**：平均精度均值
- **Precision@K**：前K个结果的精确度
- **Recall@K**：前K个结果的召回率

## 扩展功能

### 添加新的特征提取方法

1. 在 `src/` 目录下创建新的特征提取模块
2. 实现特征提取函数
3. 在 `fusion.py` 中注册新特征
4. 更新 `build_index.py` 使用新特征

### 支持新的数据集

1. 准备数据集图片
2. 修改 `build_index.py` 中的路径配置
3. 重新构建索引

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支：`git checkout -b feature/new-feature`
3. 提交更改：`git commit -am 'Add new feature'`
4. 推送分支：`git push origin feature/new-feature`
5. 提交Pull Request

## 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 致谢

- [FAISS](https://github.com/facebookresearch/faiss) - 高效的相似性搜索库
- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的Web框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue：[GitHub Issues](https://github.com/liule66/CBIR/issues)
- 邮箱：jialeliu0606@gmail.com

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

## Author
liule / [@liule66](https://github.com/liule66)
