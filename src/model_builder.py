import torch
from torch import nn
import torchvision

def create_model(num_classes: int):
    # ใช้ Weights ล่าสุดจาก ImageNet
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    model = torchvision.models.resnet50(weights=weights)

    # 1. Freeze ส่วน Feature Extractor (ไม่ต้องเทรนใหม่)
    for param in model.parameters():
        param.requires_grad = False

    # 2. ปรับเปลี่ยน Classifier Head ให้ตรงกับจำนวนคลาสสินค้าของเรา
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model