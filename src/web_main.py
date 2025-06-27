from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import faiss
from PIL import Image
import io
import os
from resnet import extract_resnet_feature  # 请确保此函数已实现

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
    feat = extract_resnet_feature(img).astype('float32').reshape(1, -1)
    D, I = index.search(feat, 5)  # 返回5个最相似
    result_imgs = [img_paths[i] for i in I[0]]
    return templates.TemplateResponse("result.html", {"request": request, "result_imgs": result_imgs}) 