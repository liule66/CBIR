import sys
import os

from resnet import extract_resnet_feature
from evaluate import evaluate_class

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import faiss
from PIL import Image
import io
import base64

app = FastAPI()

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "..", "dataset")
templates_dir = os.path.join(BASE_DIR, "..", "templates")
faiss_index_dir = os.path.join(BASE_DIR, "..", "faiss_index")

app.mount("/static", StaticFiles(directory=dataset_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# 加载faiss索引和图片路径
features = np.load(os.path.join(faiss_index_dir, 'features.npy'))
index = faiss.read_index(os.path.join(faiss_index_dir, 'index.faiss'))
with open(os.path.join(faiss_index_dir, 'img_paths.txt'), encoding='utf-8') as f:
    img_paths = [line.strip() for line in f]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # 不再保存原图到 dataset 目录
    # 直接在内存中处理图片
    # 提取特征并检索
    feat = extract_resnet_feature(img).astype('float32').reshape(1, -1)
    D, I = index.search(feat, 5)
    result_imgs = [img_paths[i] for i in I[0]]
    # 距离归一化为相似度分数（0~1，距离越小分数越高）
    d_min, d_max = float(np.min(D[0])), float(np.max(D[0]))
    if d_max > d_min:
        result_scores = [1 - (float(d) - d_min) / (d_max - d_min) for d in D[0]]
    else:
        result_scores = [1.0 for _ in D[0]]
    # 为了前端展示原图，可以将图片转为 base64 字符串传递
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    original_img_data = f"data:image/png;base64,{img_base64}"
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result_imgs": result_imgs,
            "result_scores": result_scores,
            "original_img_path": None,  # 不再传递文件名
            "original_img_data": original_img_data  # 新增字段
        }
    )


if __name__ == "__main__":
    import uvicorn
    # 让uvicorn在 0.0.0.0:8000 运行服务
    # --reload 参数可以在你修改代码后自动重启服务，非常方便
    uvicorn.run("web_main:app", host="0.0.0.0", port=8000, reload=True)