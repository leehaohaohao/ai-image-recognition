import torch
import torch.nn as nn
import timm
from safetensors.torch import load_file


def create_model():
    model = timm.create_model('efficientnet_b5', pretrained=False)

    # 使用 safetensors 加载权重
    state_dict = load_file('model.safetensors')

    # 移除分类器层的权重
    del state_dict['classifier.weight']
    del state_dict['classifier.bias']

    # 加载其余权重
    model.load_state_dict(state_dict, strict=False)

    # 重新初始化分类器层
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    return model  # 不使用 .cuda()
