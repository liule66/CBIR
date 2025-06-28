# CBIR 增量索引使用指南

## 概述

CBIR系统现在支持增量索引功能，允许您在不重新处理所有图片的情况下添加新图片到检索系统中。

## 目录结构

```
CBIR/
├── dataset/
│   ├── new/          # 新图片目录
│   └── *.jpg         # 现有图片
├── faiss_index/      # 索引文件目录
└── src/
    ├── build_index.py        # 增量索引脚本
    └── build_full_index.py   # 完整重建索引脚本
```

## 使用方法

### 1. 增量添加新图片

1. **将新图片放入 `dataset/new/` 目录**
   ```bash
   # 创建new目录（如果不存在）
   mkdir -p dataset/new
   
   # 将新图片复制到new目录
   cp your_new_images/*.jpg dataset/new/
   ```

2. **运行增量索引脚本**
   ```bash
   cd src
   python build_index.py
   ```

   脚本会自动：
   - 检测 `dataset/new/` 目录中的新图片
   - 提取特征并更新FAISS索引
   - 将新图片移动到 `dataset/` 目录（自动重命名避免冲突）
   - 更新索引中的图片路径

### 2. 完整重建索引

如果需要重新处理所有图片（例如更改特征提取方法），使用完整重建脚本：

```bash
cd src
python build_full_index.py
```

## 功能特性

### 自动文件管理
- 自动检测新图片
- 智能重命名避免文件名冲突
- 自动移动文件到主数据集目录

### 增量索引更新
- 只处理新图片，不重新处理现有图片
- 保持现有索引结构
- 高效的特征提取和索引更新

### 错误处理
- 跳过无法处理的图片
- 详细的错误日志
- 文件重命名映射记录

## 文件重命名规则

当新图片与现有图片重名时，系统会自动重命名：
- `image.jpg` → `image_1.jpg`
- `image.jpg` → `image_2.jpg`
- 以此类推

重命名映射保存在 `faiss_index/file_mapping.json` 中。

## 注意事项

1. **备份重要数据**：在首次使用前，建议备份现有的索引文件
2. **图片格式**：支持 JPG、PNG、JPEG、GIF 格式
3. **内存使用**：大量图片可能需要较多内存
4. **GPU加速**：如果安装了CUDA，ResNet和VGG特征提取会自动使用GPU

## 故障排除

### 常见问题

1. **"没有发现新图片"**
   - 确保图片已放入 `dataset/new/` 目录
   - 检查图片格式是否支持

2. **"特征提取失败"**
   - 检查图片文件是否损坏
   - 确保图片可以正常打开

3. **"加载现有索引失败"**
   - 删除损坏的索引文件，重新运行完整重建脚本
   - 检查磁盘空间是否充足

### 日志文件

- 文件重命名映射：`faiss_index/file_mapping.json`
- 控制台输出包含详细的处理信息

## 性能优化建议

1. **批量添加**：一次添加多张图片比多次添加单张图片更高效
2. **定期重建**：当新图片数量达到现有图片的50%时，考虑使用完整重建
3. **监控内存**：大量图片处理时注意内存使用情况

## 示例工作流程

```bash
# 1. 准备新图片
cp new_cat_photos/*.jpg dataset/new/

# 2. 运行增量索引
cd src
python build_index.py

# 3. 检查结果
ls dataset/ | grep cat
cat faiss_index/file_mapping.json

# 4. 启动Web服务
uvicorn web_main:app --reload --host 0.0.0.0 --port 8000
```

现在您可以通过Web界面测试新添加的图片是否可以被正确检索到！ 