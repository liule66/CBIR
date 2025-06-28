from bing_image_downloader import downloader
import os

# --- 配置区 ---
# 定义你想要下载的图片主题
keywords = [
    "dog",
]

# 为每个主题下载的图片数量
limit_per_keyword = 10

# 指定存放所有图片的总文件夹名
output_dir = 'image_library'
# --- 配置区结束 ---


# 循环下载每一个关键词
for query in keywords:
    print(f"开始下载 '{query}', 数量: {limit_per_keyword} 张...")
    downloader.download(
        query,
        limit=limit_per_keyword,
        output_dir=output_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True  # 打印详细下载过程
    )
    print(f"'{query}' 下载完成！图片保存在 '{output_dir}' 文件夹下的同名子文件夹中。\n")

print("所有下载任务已完成！")