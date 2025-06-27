from PIL import Image

def preprocess_image(img: Image.Image, size=(224, 224)):
    return img.resize(size) 