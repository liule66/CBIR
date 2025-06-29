[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source-150x25.png?v=103)](https://github.com/ellerbrock/open-source-badges/)

# CBIR (Content-Based Image Retrieval) 系统

基于内容的图像检索系统，使用深度学习特征和FAISS向量索引实现高效的相似图像搜索。

## 项目简介

本项目实现了一个完整的基于内容的图像检索系统，具有以下特点：

- **多种特征提取方法**：支持ResNet、VGG、颜色直方图、HOG等多种特征
- **高效向量检索**：使用FAISS进行快速相似性搜索
- **Web界面**：基于FastAPI的现代化Web界面
- **增量索引**：支持动态添加新图片，无需重新构建整个索引
- **🔥 交互式检索**：用户可选择特定特征类型进行检索
- **模块化设计**：易于扩展和维护

## 功能特性

- 🖼️ 支持多种图像格式（JPG、PNG、JPEG、GIF）
- 🔍 基于深度学习的图像特征提取
- ⚡ 高效的向量相似性搜索
- 🌐 现代化的Web用户界面
- 📊 支持批量处理和索引构建
- 🔄 **增量索引功能**：动态添加新图片，无需重新处理所有图片
- 🎯 **交互式检索功能**：用户可选择特定特征类型进行检索
- 🔧 可配置的特征提取参数
- 📁 智能文件管理：自动重命名和移动文件

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
# 首次构建完整索引
python src/build_full_index.py
# 或者使用增量索引（推荐）
python src/build_index.py
```

5. **启动服务**
```bash
cd src
uvicorn web_main:app --reload --host 0.0.0.0 --port 8000
```

6. **访问系统**
打开浏览器访问：http://localhost:8000

## 项目结构

```
CBIR/
├── src/                    # 源代码目录
│   ├── web_main.py        # FastAPI主应用
│   ├── build_index.py     # 增量索引构建脚本
│   ├── build_full_index.py # 完整索引重建脚本
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
│   ├── index.html         # 主页（支持交互式检索）
│   └── result.html        # 结果页
├── dataset/               # 图片数据集
│   └── new/               # 新图片目录（增量索引用）
├── faiss_index/           # FAISS索引文件
├── cache/                 # 缓存文件
├── result/                # 评估结果
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明
├── INCREMENTAL_USAGE.md  # 增量索引使用指南
├── demo_interactive_search.py # 交互式检索演示脚本
└── USAGE.md              # 使用指南
```

## 使用方法

### 1. 准备数据集

将图片文件放入 `dataset/` 目录，支持JPG、PNG、JPEG、GIF格式。

### 2. 构建索引

#### 首次构建（完整索引）
```bash
python src/build_full_index.py
```

#### 增量索引（推荐）
```bash
python src/build_index.py
```

这将：
- 遍历 `dataset/` 目录中的所有图片
- 提取多种特征（ResNet、VGG、颜色、纹理、形状等）
- 构建FAISS索引
- 保存索引文件到 `faiss_index/` 目录

### 3. 增量索引（新增图片）

当有新图片需要添加到检索系统时：

1. **将新图片放入 `dataset/new/` 目录**
```bash
mkdir -p dataset/new
# 将新图片复制到 dataset/new/ 目录
```

2. **运行增量索引**
```bash
python src/build_index.py
```

增量索引功能会：
- 自动检测 `dataset/new/` 目录中的新图片
- 提取特征并更新FAISS索引
- 将新图片移动到 `dataset/` 目录（自动重命名避免冲突）
- 更新索引中的图片路径
- 保存文件重命名映射到 `faiss_index/file_mapping.json`

### 4. 启动Web服务

```bash
cd src
uvicorn web_main:app --reload --host 0.0.0.0 --port 8000
```

### 5. 使用交互式检索功能

#### 🎯 交互式检索特性

1. **特征选择**：用户可以选择一种或多种特征类型进行检索
2. **全选功能**：提供"全选/取消全选"快捷按钮
3. **智能保护**：确保至少选择一个特征类型，防止误操作
4. **结果展示**：只显示用户选择的特征类型的检索结果
5. **特征标识**：在结果页面显示本次检索使用的特征类型

#### 📋 可选择的特征类型

- **颜色特征** (Color Features)：基于颜色直方图的检索
- **纹理特征** (Texture Features)：基于DAISY纹理描述符的检索
- **形状特征** (Shape Features)：基于HOG梯度方向直方图的检索
- **ResNet特征** (Deep Learning)：基于深度学习的ResNet特征
- **VGG特征** (Deep Learning)：基于深度学习的VGG特征
- **融合特征** (Fusion Features)：综合多种特征的融合检索

#### 🔧 使用步骤

1. 打开浏览器访问：http://localhost:8000
2. 上传一张图片
3. 在特征选择区域选择需要的特征类型：
   - 勾选复选框选择特定特征
   - 使用"全选/取消全选"按钮快速操作
   - 系统会确保至少选择一个特征类型
4. 点击"开始检索"按钮
5. 查看只包含所选特征的检索结果

#### 💡 使用建议

- **颜色相似检索**：选择"颜色特征"
- **纹理相似检索**：选择"纹理特征"
- **形状相似检索**：选择"形状特征"
- **精确检索**：选择深度学习特征（ResNet/VGG）
- **综合检索**：选择"融合特征"
- **全面检索**：使用"全选"功能

## 增量索引详细说明

### 优势
- **高效**：只处理新图片，不重新处理现有图片
- **智能**：自动文件管理和重命名
- **安全**：保持索引一致性，避免数据丢失
- **灵活**：支持批量添加和单张添加

### 工作流程
1. 检测 `dataset/new/` 目录中的新图片
2. 提取新图片的特征
3. 更新现有FAISS索引
4. 移动新图片到主数据集目录
5. 更新索引中的图片路径
6. 保存文件重命名映射

### 文件重命名规则
当新图片与现有图片重名时，系统会自动重命名：
- `image.jpg` → `image_1.jpg`
- `image.jpg` → `image_2.jpg`
- 以此类推

详细使用说明请参考 [INCREMENTAL_USAGE.md](INCREMENTAL_USAGE.md)

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

## 扩展功能

### 添加新的特征提取方法

1. 在 `src/` 目录下创建新的特征提取模块
2. 实现特征提取函数
3. 在 `fusion.py` 中注册新特征
4. 更新 `build_index.py` 使用新特征
5. 在 `web_main.py` 中添加新特征到 `feature_methods` 字典
6. 在 `templates/index.html` 中添加新特征的复选框

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
