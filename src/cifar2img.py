import os
import pickle
import numpy as np
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images_from_batch(batch_file, save_dir):
    batch = unpickle(batch_file)
    data = batch[b'data']
    labels = batch[b'labels']
    filenames = batch[b'filenames']
    os.makedirs(save_dir, exist_ok=True)
    for i, img_flat in enumerate(data):
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)  # (32,32,3)
        img = Image.fromarray(img)
        label = labels[i]
        fname = filenames[i].decode('utf-8')
        img.save(os.path.join(save_dir, f"{label}_{fname}"))

if __name__ == "__main__":
    cifar_dir = "dataset/cifar-10-batches-py"
    save_dir = "dataset/cifar10_images"
    batch_files = [f for f in os.listdir(cifar_dir) if f.startswith("data_batch") or f.startswith("test_batch")]
    for batch_file in batch_files:
        save_images_from_batch(os.path.join(cifar_dir, batch_file), save_dir)
    print("图片已全部保存到", save_dir)