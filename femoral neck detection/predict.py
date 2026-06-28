from ultralytics import YOLO
import torch
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from utils import calculate_mapping

# ========= 配置部分 =========

model = YOLO("best.pt")  # 训练好的YOLO11权重

# ===========================

# 输入输出路径

calculate_mapping = calculate_mapping()
device = "cuda" if torch.cuda.is_available() else "cpu"
laterality_dict = calculate_mapping.load_laterality("data/mapping rule.xlsx", col_seq='image_name')
image_dir = Path("")
out_dir = Path("data/crop_side")
out_dir.mkdir(parents=True, exist_ok=True)

#裁剪参数
pad_ratio = 0.2
conf_thres = 0.25
results = model.predict(source=str(image_dir), conf=conf_thres, imgsz=1024, stream=True) #1024是模型推理的输入尺寸

ious = []
dices = []

processed_count = 0
no_laterality_count = 0
no_boxes_count = 0

for r in results:

    img_path = Path(r.path)
    img_name = img_path.stem
    
    img = cv2.imread(str(r.path))
    if img is None:
        continue
    h, w = img.shape[:2]
    
    boxes = r.boxes
    if boxes is None or len( boxes) == 0:
        continue
    
    if img_name not in laterality_dict:
        print('该图片为侧位或仅有一侧关节')
        # 只取top1，按照置信度排序
        confs = boxes.conf.cpu().numpy()
        best_i = confs.argmax()
        xyxy = boxes.xyxy[best_i].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        pred_box = [x1, y1, x2, y2]
    
        # padding，裁剪并保存
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
    
        nx1 = max(0, int(x1 - px))
        ny1 = max(0, int(y1 - py))
        nx2 = min(w - 1, int(x2 + px))
        ny2 = min(h - 1, int(y2 + py))
    
        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            continue
    
        save_path = out_dir / (Path(r.path).stem + ".jpg")
        cv2.imwrite(str(save_path), crop)
    
        # # 读取GT（真实标注框）标签
        # label_path = Path(str(r.path)
        #                   .replace("images", "labels")
        #                   .replace(".jpg", ".txt"))
        #
        # gt_boxes = []
        # if label_path.exists():
        #     with open(label_path, "r") as f:
        #         for line in f:
        #             cls, cx, cy, bw, bh = map(float, line.split())
        #             gx1 = (cx - bw / 2) * w
        #             gy1 = (cy - bh / 2) * h
        #             gx2 = (cx + bw / 2) * w
        #             gy2 = (cy + bh / 2) * h
        #             gt_boxes.append([gx1, gy1, gx2, gy2])
        #
        # # 计算IOU和Dice
        # best_iou = 0
        # best_dice = 0
        # for gt_box in gt_boxes:
        #     iou = calculate_mapping.calc_iou(pred_box, gt_box)
        #     dice = calculate_mapping.calc_dice(pred_box, gt_box)
        #     if iou > best_iou:
        #         best_iou = iou
        #         best_dice = dice
        #
        # ious.append(best_iou)
        # dices.append(best_dice)
        #
        # print(f"{Path(r.path).name} | IoU: {best_iou:.4f} | Dice: {best_dice:.4f}")
    
    else:
        image_laterality = laterality_dict[img_name]
        print(f"  影像侧别: {image_laterality}")
        crop, pred_box = calculate_mapping.select_and_crop_by_image_laterality(
            img, boxes, image_laterality, w, pad_ratio
        )
        save_path = out_dir / (Path(r.path).stem + ".jpg")
        cv2.imwrite(str(save_path), crop)
    
        # # 读取GT（真实标注框）标签
        # label_path = Path(str(r.path)
        #                   .replace("images", "labels")
        #                   .replace(".jpg", ".txt"))
        #
        # gt_boxes = []
        # if label_path.exists():
        #     with open(label_path, "r") as f:
        #         for line in f:
        #             cls, cx, cy, bw, bh = map(float, line.split())
        #             gx1 = (cx - bw / 2) * w
        #             gy1 = (cy - bh / 2) * h
        #             gx2 = (cx + bw / 2) * w
        #             gy2 = (cy + bh / 2) * h
        #             gt_boxes.append([gx1, gy1, gx2, gy2])
        #
        # # 计算IOU和Dice
        # best_iou = 0
        # best_dice = 0
        # for gt_box in gt_boxes:
        #     iou = calculate_mapping.calc_iou(pred_box, gt_box)
        #     dice = calculate_mapping.calc_dice(pred_box, gt_box)
        #     if iou > best_iou:
        #         best_iou = iou
        #         best_dice = dice
        #
        # ious.append(best_iou)
        # dices.append(best_dice)
        #
        # print(f"{Path(r.path).name} | IoU: {best_iou:.4f} | Dice: {best_dice:.4f}")

print("\n========== Validation Result ==========")
print(f"Mean IoU : {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
