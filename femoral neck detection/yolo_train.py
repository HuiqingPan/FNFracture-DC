from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n.pt')  # 加载预训练模型

# 使用自定义配置文件训练
results = model.train(
    data= 'dataset.yaml',  # 配置文件路径（如果不在同目录需写全路径）
    epochs=80,
    imgsz=768,
    batch=16,
    device=-1,  # 0表示使用GPU，-1表示使用CPU
    lr0=0.005,
    lrf=0.01,
    weight_decay=0.0005,  # 权重衰减，默认0.0005，可尝试0.001
    patience=30,
    warmup_epochs=2,  # 学习率预热轮数
    mosaic=1.0,  # Mosaic数据增强概率，默认1.0
    mixup=0.1,  # Mixup数据增强概率，默认0.0，建议0.1-0.15
    close_mosaic=10,  # 最后N轮关闭mosaic，默认10
)
