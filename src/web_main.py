import sys
import os
import numpy as np
import faiss
from PIL import Image
import io
import base64

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 导入特征提取方法
from color import Color
from daisy import Daisy
from HOG import HOG
from resnet import extract_resnet_feature
from vggnet import extract_vgg_feature
from fusion import extract_fusion_feature  # 你可以自定义

sys.path.insert(0, os.path.dirname(__file__))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# 使用绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "..", "dataset")
templates_dir = os.path.join(BASE_DIR, "..", "templates")
faiss_index_dir = os.path.join(BASE_DIR, "..", "faiss_index")

app.mount("/static", StaticFiles(directory=dataset_dir), name="static")
app.mount("/material", StaticFiles(directory=os.path.join(BASE_DIR, "..", "material")), name="material")
templates = Jinja2Templates(directory=templates_dir)

# 统一key为英文小写
feature_methods = {
    "color": lambda img: Color().histogram(np.array(img)),
    "texture": lambda img: Daisy().histogram(np.array(img)),
    "shape": lambda img: HOG().histogram(np.array(img)),
    "resnet": extract_resnet_feature,
    "vgg": extract_vgg_feature,
    "fusion": extract_fusion_feature,
}
feature_names = {
    "color": "颜色特征",
    "texture": "纹理特征",
    "shape": "形状特征",
    "resnet": "ResNet特征",
    "vgg": "VGG特征",
    "fusion": "融合特征",
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # 转为 base64 以便前端展示原图
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    original_img_data = f"data:image/png;base64,{img_base64}"

    all_results = []
    for feature_type, extract_func in feature_methods.items():
        try:
            feat = np.array(extract_func(img)).astype('float32').reshape(1, -1)
            features = np.load(os.path.join(faiss_index_dir, f'features_{feature_type}.npy'))
            index = faiss.read_index(os.path.join(faiss_index_dir, f'index_{feature_type}.faiss'))
            with open(os.path.join(faiss_index_dir, f'img_paths_{feature_type}.txt'), encoding='utf-8') as f:
                img_paths = [line.strip() for line in f]
            D, I = index.search(feat, 5)
            result_imgs = [img_paths[i] for i in I[0]]
            d_min, d_max = float(np.min(D[0])), float(np.max(D[0]))
            if d_max > d_min:
                result_scores = [1 - (float(d) - d_min) / (d_max - d_min) for d in D[0]]
            else:
                result_scores = [1.0 for _ in D[0]]
            all_results.append({
                "feature_name": feature_names[feature_type],
                "result_imgs": result_imgs,
                "result_scores": result_scores
            })
        except Exception as e:
            all_results.append({
                "feature_name": feature_names[feature_type],
                "result_imgs": [],
                "result_scores": [],
                "error": str(e)
            })

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "original_img_data": original_img_data,
            "all_results": all_results
        }
    )

if __name__ == "__main__":
    import uvicorn
    # 让uvicorn在 0.0.0.0:8000 运行服务
    # --reload 参数可以在你修改代码后自动重启服务，非常方便
    uvicorn.run("web_main:app", host="0.0.0.0", port=8000, reload=True)