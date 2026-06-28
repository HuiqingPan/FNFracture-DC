from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
import numpy as np

# ========= 配置部分 =========
model = YOLO("best.pt")  # 训练好的YOLO11权重
# ===========================

# 使用自定义配置文件验证
results = model.val(
    data= 'dataset_external.yaml',  # 配置文件路径（如果不在同目录需写全路径）
    imgsz=768,
    batch=9,
    save_json=True,
    device=-1,  # 0表示使用GPU，-1表示使用CPU
)

# 所有 IoU 阈值下的 mAP
maps =results.box.map75
f1_score = results.box.f1
print(f"mAP: {maps.mean():.4f} | f1:{f1_score.mean():.4f}")

print(dir(results.box))
print(results.results_dict)

print(results.box.nc)