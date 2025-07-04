# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo

from six.moves import cPickle
import numpy as np
import scipy.misc
import os
from PIL import Image
import imageio
import gc

from evaluate import evaluate_class
from DB import Database

os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'cache')

'''
  downloading problem in mac OSX should refer to this answer:
    https://stackoverflow.com/a/42334357
'''

# configs for histogram
RES_model  = 'resnet152'  # model type
pick_layer = 'avg'        # extract feature of this layer
d_type     = 'd1'         # distance type

depth = 3  # retrieved depth, set to None will count the ap for whole database

''' MMAP
     depth
      depthNone, resnet152,avg,d1, MMAP 0.78474710149
      depth100,  resnet152,avg,d1, MMAP 0.819713653589
      depth30,   resnet152,avg,d1, MMAP 0.884925001919
      depth10,   resnet152,avg,d1, MMAP 0.944355078125
      depth5,    resnet152,avg,d1, MMAP 0.961788675194
      depth3,    resnet152,avg,d1, MMAP 0.965623938039
      depth1,    resnet152,avg,d1, MMAP 0.958696281702

      (exps below use depth=None)

      resnet34,avg,cosine, MMAP 0.755842698037
      resnet101,avg,cosine, MMAP 0.757435452078
      resnet101,avg,d1, MMAP 0.764556148137
      resnet152,avg,cosine, MMAP 0.776918319273
      resnet152,avg,d1, MMAP 0.78474710149
      resnet152,max,d1, MMAP 0.748099342614
      resnet152,fc,cosine, MMAP 0.776918319273
      resnet152,fc,d1, MMAP 0.70010267663
'''

use_gpu = torch.cuda.is_available()
means = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
  os.makedirs(cache_dir)

# 全局模型实例，避免重复创建
_resnet_model = None

def get_resnet_model():
    """获取ResNet模型实例（单例模式）"""
    global _resnet_model
    if _resnet_model is None:
        _resnet_model = ResidualNet(model=RES_model)
        _resnet_model.eval()
        if use_gpu:
            _resnet_model = _resnet_model.cuda()
    return _resnet_model

def clear_resnet_model():
    """清理ResNet模型实例"""
    global _resnet_model
    if _resnet_model is not None:
        del _resnet_model
        _resnet_model = None
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResidualNet(ResNet):
  def __init__(self, model=RES_model, pretrained=True):
    if model == "resnet18":
        super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    elif model == "resnet34":
        super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    elif model == "resnet50":
        super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    elif model == "resnet101":
        super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    elif model == "resnet152":
        super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
        if pretrained:
            self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
    max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
    Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
    Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
    avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    fc = self.fc(avg)  # fc.size = N * 1000
    output = {
        'max': Max,
        'avg': avg,
        'fc' : fc
    }
    return output


class ResNetFeat(object):

  def make_samples(self, db, verbose=True):
    sample_cache = '{}-{}'.format(RES_model, pick_layer)
  
    try:
      samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
      for sample in samples:
        sample['hist'] /= np.sum(sample['hist'])  # normalize
      if verbose:
        print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
    except:
      if verbose:
        print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
  
      res_model = ResidualNet(model=RES_model)
      res_model.eval()
      if use_gpu:
        res_model = res_model.cuda()
      samples = []
      data = db.get_data()
      for d in data.itertuples():
        d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
        img = imageio.imread(d_img, mode="RGB")
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= means[0]  # reduce B's mean
        img[1] -= means[1]  # reduce G's mean
        img[2] -= means[2]  # reduce R's mean
        img = np.expand_dims(img, axis=0)
        try:
          if use_gpu:
            inputs = torch.autograd.Variable(torch.from_numpy(img).cuda().float())
          else:
            inputs = torch.autograd.Variable(torch.from_numpy(img).float())
          d_hist = res_model(inputs)[pick_layer]
          d_hist = d_hist.data.cpu().numpy().flatten()
          d_hist /= np.sum(d_hist)  # normalize
          samples.append({
                          'img':  d_img, 
                          'cls':  d_cls, 
                          'hist': d_hist
                         })
        except:
          pass
      cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))
  
    return samples


def extract_resnet_feature(img):
    """
    从PIL Image对象提取ResNet特征
    Args:
        img: PIL Image对象
    Returns:
        feature: numpy array, shape (512,)
    """
    try:
        # 获取模型实例（单例模式）
        res_model = get_resnet_model()
        
        # 预处理图片
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        img_array = np.transpose(img_array, (2, 0, 1)) / 255.
        img_array[0] -= means[0]  # reduce B's mean
        img_array[1] -= means[1]  # reduce G's mean
        img_array[2] -= means[2]  # reduce R's mean
        img_array = np.expand_dims(img_array, axis=0)
        
        # 提取特征
        with torch.no_grad():  # 禁用梯度计算以节省内存
            if use_gpu:
                inputs = torch.from_numpy(img_array).cuda().float()
            else:
                inputs = torch.from_numpy(img_array).float()
            
            feature = res_model(inputs)[pick_layer]
            feature = feature.cpu().numpy().flatten()
            feature /= np.sum(feature)  # normalize
            
            # 清理中间变量
            del inputs
            if use_gpu:
                torch.cuda.empty_cache()
            
            return feature
    except Exception as e:
        print(f"Error extracting ResNet feature: {e}")
        # 清理内存
        if use_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        return None


if __name__ == "__main__":
  # evaluate database
  db = Database()
  APs = evaluate_class(db, f_class=ResNetFeat, d_type=d_type, depth=depth)
  cls_MAPs = []
  for cls, cls_APs in APs.items():
    MAP = np.mean(cls_APs)
    print("Class {}, MAP {}".format(cls, MAP))
    cls_MAPs.append(MAP)
  print("MMAP", np.mean(cls_MAPs))
