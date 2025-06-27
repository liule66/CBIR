from simple_image_download import simple_image_download as simp

# 实例化下载器
response = simp.simple_image_download

# 定义你要下载的图片主题和数量
# 格式：{"关键词": 数量}
keywords_and_limits = {
    "Mount Fuji": 80,         # 下载80张关于“富士山”的图片
    "Eiffel Tower": 80,       # 下载80张关于“埃菲尔铁塔”的图片
    "Golden Retriever puppy": 80, # 下载80张关于“金毛幼犬”的图片
    "mygo": 80,
}

# 循环下载
for keyword, limit in keywords_and_limits.items():
    print(f"开始下载 '{keyword}', 数量: {limit} 张...")
    response().download(keywords=keyword, limit=limit)
    print(f"'{keyword}' 下载完成！\n")

print("所有图片已下载到 simple_images 文件夹中。")