import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms



def _read_csv_with_fallback(file_path):
    """Read a CSV file with common encodings used by the project files."""
    last_error = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(file_path, encoding=enc), enc
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def read_file(file_path, image_col="path"):
    """
    Read a split CSV and keep row-level image records.

    The downstream dataset groups these rows by patient ``id``. A patient can have
    one or two valid image rows. Labels are patient-level and are recovered as the
    first non-missing value within each ``id`` group, so rows for a second view may
    leave labels blank.
    """
    df, used_encoding = _read_csv_with_fallback(file_path)
    required = ["id", "fracture"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path} 缺少必要列: {missing}")
    if image_col not in df.columns:
        raise ValueError(
            f"{file_path} 缺少图像路径列 '{image_col}'。"
            f"当前列为: {list(df.columns)}；可用 --image-col 指定实际图像路径列。"
        )
    
    df = df.copy()
    df["_csv_encoding"] = used_encoding
    df["_row_order"] = np.arange(len(df))
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    
    for col in ["fracture", "garden", "angle"]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

def build_label_map(*dfs, column):
    values = []
    for df in dfs:
        if column in df.columns:
            values.extend(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).tolist())
    values = sorted(set(values))
    if not values:
        raise ValueError(f"没有找到有效的 {column} 标签，无法建立分类头。")
    return {label: idx for idx, label in enumerate(values)}


def infer_num_classes(*dfs, column, minimum=2):
    values = []
    for df in dfs:
        if column in df.columns:
            values.extend(pd.to_numeric(df[column], errors="coerce").dropna().astype(int).tolist())
    if not values:
        return minimum
    return max(minimum, int(max(values)) + 1)


class ResizePad:
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new(img.mode, (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas



def _has_valid_path(value):
    if pd.isna(value):
        return False
    value = str(value).strip()
    return value != "" and value.lower() not in {"nan", "none", "null"}


def _first_valid(series, default=np.nan):
    for value in series:
        if pd.notna(value):
            return value
    return default


def _join_valid(values, sep=" | "):
    out = []
    for value in values:
        if pd.notna(value) and str(value).strip() != "":
            out.append(str(value))
    return sep.join(out)


def _view_sort_key(row):
    """Sort AP-like rows before lateral-like rows when possible, else use suffix/order."""
    for col in ["view", "projection", "view_type", "image_view"]:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip().lower()
            if any(k in text for k in ["ap", "anteroposterior", "front", "正"]):
                return (0, int(row.get("_row_order", 0)))
            if any(k in text for k in ["lat", "lateral", "side", "侧"]):
                return (1, int(row.get("_row_order", 0)))

    for col in ["source_id", "source_row", "raw_id"]:
        if col in row.index and pd.notna(row[col]):
            match = re.search(r"_(\d+)$", str(row[col]).strip())
            if match:
                return (int(match.group(1)), int(row.get("_row_order", 0)))
    return (0, int(row.get("_row_order", 0)))


def make_patient_level_df(df, image_col="path", max_views=2):
    """
    Convert row-level image records to patient-level samples.

    The first valid view is treated as AP/primary view; the second valid view is
    treated as lateral/auxiliary view. If a patient has only one valid image path,
    cross-attention is bypassed for that sample.
    """
    patient_rows = []
    skipped_no_image = 0
    skipped_no_label = 0
    
    for patient_id, group in df.groupby("id", sort=False):
        group = group.copy()
        if "_row_order" not in group.columns:
            group["_row_order"] = np.arange(len(group))
        group["_view_sort"] = group.apply(_view_sort_key, axis=1)
        group = group.sort_values("_view_sort", kind="stable")
    
        valid = group[group[image_col].apply(_has_valid_path)].copy()
        if len(valid) == 0:
            skipped_no_image += 1
            continue
    
        fracture = _first_valid(group["fracture"])
        if pd.isna(fracture):
            skipped_no_label += 1
            continue
    
        view_rows = valid.head(max_views)
        view_paths = [str(v).strip() for v in view_rows[image_col].tolist()]
    
        row = {
            "id": int(patient_id),
            "fracture": int(fracture),
            "garden": _first_valid(group["garden"]),
            "angle": _first_valid(group["angle"]),
            "n_views": int(len(view_paths)),
            "has_lateral": bool(len(view_paths) >= 2),
            "path": view_paths[0],
            "view_paths": _join_valid(view_paths),
            "_view_paths": view_paths,
        }
    
        for col in ["source_id", "source_row", "raw_id", "source_image", "injury_side", "split", "md5", "pauwels"]:
            if col in group.columns:
                if col in ["source_id", "source_row", "raw_id", "source_image", "md5"]:
                    row[col] = _join_valid(view_rows[col].tolist())
                else:
                    row[col] = _first_valid(group[col])
    
        patient_rows.append(row)
    
    out = pd.DataFrame(patient_rows)
    if len(out) == 0:
        raise ValueError("没有可用的病人级样本：请检查 id、图像路径列和 fracture 标签。")
    if skipped_no_image:
        print(f"警告：{skipped_no_image} 个 id 没有有效图像路径，已跳过。")
    if skipped_no_label:
        print(f"警告：{skipped_no_label} 个 id 没有 fracture 标签，已跳过。")
    return out.reset_index(drop=True)


class PatientXrayDataset(Dataset):
    def __init__(
        self,
        df,
        transform,
        garden_label_map,
        image_col="path",
        max_views=2,
        train=False,
        shared_hflip_p=0.0,
    ):
        self.df = make_patient_level_df(df, image_col=image_col, max_views=max_views)
        self.transform = transform
        self.garden_label_map = garden_label_map
        self.train = train
        self.shared_hflip_p = float(shared_hflip_p)

    def __len__(self):
        return len(self.df)
    
    @staticmethod
    def _load_image(path):
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)
        return img
    
    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        view_paths = list(r["_view_paths"])
        img_ap = self._load_image(view_paths[0])
        has_lateral = len(view_paths) >= 2
        img_lat = self._load_image(view_paths[1]) if has_lateral else None
    
        # Apply geometric hflip jointly so paired AP/lateral correspondence is not broken.
        if self.train and self.shared_hflip_p > 0 and torch.rand(()) < self.shared_hflip_p:
            img_ap = ImageOps.mirror(img_ap)
            if img_lat is not None:
                img_lat = ImageOps.mirror(img_lat)
    
        x_ap = self.transform(img_ap)
        if has_lateral:
            x_lat = self.transform(img_lat)
        else:
            x_lat = torch.zeros_like(x_ap)
    
        fracture = int(r["fracture"])
    
        if pd.isna(r["garden"]):
            garden = -100
        else:
            garden_raw = int(r["garden"])
            if garden_raw not in self.garden_label_map:
                raise ValueError(f"未知 garden 标签 {garden_raw}，请检查训练/验证/测试数据标签是否一致。")
            garden = self.garden_label_map[garden_raw]
    
        angle = float("nan") if pd.isna(r["angle"]) else float(r["angle"])
    
        return (
            x_ap,
            x_lat,
            torch.tensor(has_lateral, dtype=torch.bool),
            torch.tensor(fracture, dtype=torch.long),
            torch.tensor(garden, dtype=torch.long),
            torch.tensor(angle, dtype=torch.float32),
            torch.tensor(int(r["id"]), dtype=torch.long),
            torch.tensor(int(r["n_views"]), dtype=torch.long),
        )


# Backward-compatible alias for any external code that imports XrayDataset.

XrayDataset = PatientXrayDataset


class MultiViewCrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-view attention for patient-level AP/lateral features.

    The CNN backbone produces one vector per view. To keep the module lightweight,
    attention is performed over projected feature channels. Paired samples are
    fused as: AP attends to lateral, lateral attends to AP, residual + LayerNorm,
    concatenation, and projection back to the backbone feature dimension so the
    original task heads can be reused. Single-view samples bypass this module.
    """
    
    def __init__(self, features, attn_dim=0, dropout=0.0):
        super().__init__()
        self.features = int(features)
        self.attn_dim = int(attn_dim) if int(attn_dim) > 0 else int(features)
        scale_dim = max(self.attn_dim, 1)
        self.scale = math.sqrt(scale_dim)
    
        self.ap_q = nn.Linear(features, self.attn_dim)
        self.ap_k = nn.Linear(features, self.attn_dim)
        self.ap_v = nn.Linear(features, self.attn_dim)
        self.lat_q = nn.Linear(features, self.attn_dim)
        self.lat_k = nn.Linear(features, self.attn_dim)
        self.lat_v = nn.Linear(features, self.attn_dim)
    
        self.ap_out = nn.Linear(self.attn_dim, features)
        self.lat_out = nn.Linear(self.attn_dim, features)
        self.ap_norm = nn.LayerNorm(features)
        self.lat_norm = nn.LayerNorm(features)
        self.drop = nn.Dropout(dropout)
        self.fuse = nn.Sequential(
            nn.LayerNorm(features * 2),
            nn.Dropout(dropout),
            nn.Linear(features * 2, features),
            nn.ReLU(inplace=True),
            nn.LayerNorm(features),
        )
    
    def _channel_attention(self, q, k, v):
        weights = torch.softmax((q * k) / self.scale, dim=-1)
        return weights * v
    
    def forward(self, ap_feats, lat_feats):
        ap_context = self._channel_attention(self.ap_q(ap_feats), self.lat_k(lat_feats), self.lat_v(lat_feats))
        lat_context = self._channel_attention(self.lat_q(lat_feats), self.ap_k(ap_feats), self.ap_v(ap_feats))
    
        refined_ap = self.ap_norm(ap_feats + self.drop(self.ap_out(ap_context)))
        refined_lat = self.lat_norm(lat_feats + self.drop(self.lat_out(lat_context)))
        fused = self.fuse(torch.cat([refined_ap, refined_lat], dim=1))
        return fused


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        arch,
        pretrained=True,
        head_hidden=0,
        dropout=0.0,
        fracture_classes=2,
        garden_classes=2,
        use_cross_attention=True,
        cross_attn_dim=0,
    ):
        super().__init__()
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            net = models.resnet18(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            net = models.resnet34(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "efficientnet_v2_s":
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_v2_s(weights=weights)
            features = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif arch == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            net = models.convnext_tiny(weights=weights)
            features = net.classifier[2].in_features
            net.classifier = nn.Identity()
        elif arch == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v2(weights=weights)
            features = net.classifier[1].in_features
            net.classifier = nn.Identity()
        elif arch == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_small(weights=weights)
            features = net.classifier[3].in_features
            net.classifier = nn.Identity()
        elif arch == "mobilenet_v3_large":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            net = models.mobilenet_v3_large(weights=weights)
            features = net.classifier[3].in_features
            net.classifier = nn.Identity()
        elif arch == "shufflenet_v2_x0_5":
            weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
            net = models.shufflenet_v2_x0_5(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "shufflenet_v2_x1_0":
            weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
            net = models.shufflenet_v2_x1_0(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "shufflenet_v2_x1_5":
            weights = models.ShuffleNet_V2_X1_5_Weights.DEFAULT if pretrained else None
            net = models.shufflenet_v2_x1_5(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        elif arch == "shufflenet_v2_x2_0":
            weights = models.ShuffleNet_V2_X2_0_Weights.DEFAULT if pretrained else None
            net = models.shufflenet_v2_x2_0(weights=weights)
            features = net.fc.in_features
            net.fc = nn.Identity()
        else:
            raise ValueError(arch)

        self.backbone = net
        self.features = features
        self.use_cross_attention = bool(use_cross_attention)
        self.cross_attention = (
            MultiViewCrossAttentionFusion(features, attn_dim=cross_attn_dim, dropout=dropout)
            if self.use_cross_attention
            else None
        )
        self.angle_head = self._make_head(features, 1, head_hidden, dropout)
        self.fracture_head = self._make_head(features, fracture_classes, head_hidden, dropout)
        self.garden_head = self._make_head(features, garden_classes, head_hidden, dropout)
    
    @staticmethod
    def _make_head(features, outputs, head_hidden, dropout):
        if head_hidden <= 0:
            return nn.Linear(features, outputs)
        return nn.Sequential(
            nn.LayerNorm(features),
            nn.Dropout(dropout),
            nn.Linear(features, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, outputs),
        )
    
    def _encode_and_fuse(self, x_ap, x_lat=None, has_lateral=None):
        ap_feats = self.backbone(x_ap)
        if not self.use_cross_attention or x_lat is None or has_lateral is None:
            return ap_feats
    
        has_lateral = has_lateral.bool().view(-1)
        if not torch.any(has_lateral):
            return ap_feats
    
        fused_feats = ap_feats.clone()
        lat_feats = self.backbone(x_lat[has_lateral])
        fused_pair_feats = self.cross_attention(ap_feats[has_lateral], lat_feats)
        fused_feats[has_lateral] = fused_pair_feats
        return fused_feats
    
    def forward(self, x_ap, x_lat=None, has_lateral=None):
        feats = self._encode_and_fuse(x_ap, x_lat=x_lat, has_lateral=has_lateral)
        angle_pred = self.angle_head(feats).squeeze(1)
        fracture_logits = self.fracture_head(feats)
        garden_logits = self.garden_head(feats)
        return angle_pred, fracture_logits, garden_logits


def make_transform(size, train, photometric_aug=False, random_hflip=False):
    ops = [ResizePad(size)]
    if train:
        # Do not rotate or anisotropically resize: the fracture angle is defined
        # against the image horizontal line, so those transforms corrupt labels.
        # For paired samples, horizontal flipping is handled jointly in the Dataset.
        if random_hflip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        if photometric_aug:
            ops.extend(
                [
                    transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.18)], p=0.7),
                    transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.25),
                ]
            )
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)

def finite_or_nan(value):
    try:
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def add_regression_metrics(metrics, prefix, true, pred):
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true = true[mask]
    pred = pred[mask]

    metrics[f"{prefix}_n"] = int(len(true))
    if len(true) == 0:
        for k in ["mae", "rmse", "r2", "pearson_r", "bias", "sd_diff", "loa_low", "loa_high"]:
            metrics[f"{prefix}_{k}"] = float("nan")
        return
    
    diff = pred - true
    sd_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan")
    pearson = stats.pearsonr(true, pred).statistic if len(true) > 1 else float("nan")
    r2 = r2_score(true, pred) if len(true) > 1 else float("nan")
    
    metrics.update(
        {
            f"{prefix}_mae": float(mean_absolute_error(true, pred)),
            f"{prefix}_rmse": float(math.sqrt(mean_squared_error(true, pred))),
            f"{prefix}_r2": finite_or_nan(r2),
            f"{prefix}_pearson_r": finite_or_nan(pearson),
            f"{prefix}_bias": float(np.mean(diff)),
            f"{prefix}_sd_diff": sd_diff,
            f"{prefix}_loa_low": float(np.mean(diff) - 1.96 * sd_diff) if np.isfinite(sd_diff) else float("nan"),
            f"{prefix}_loa_high": float(np.mean(diff) + 1.96 * sd_diff) if np.isfinite(sd_diff) else float("nan"),
        }
    )


def add_classification_metrics(metrics, prefix, true, pred):
    true = np.asarray(true)
    pred = np.asarray(pred)
    mask = np.isfinite(true.astype(float)) & np.isfinite(pred.astype(float))
    true = true[mask].astype(int)
    pred = pred[mask].astype(int)

    metrics[f"{prefix}_n"] = int(len(true))
    if len(true) == 0:
        metrics[f"{prefix}_acc"] = float("nan")
        metrics[f"{prefix}_f1_macro"] = float("nan")
        return
    
    metrics[f"{prefix}_acc"] = float(accuracy_score(true, pred))
    metrics[f"{prefix}_f1_macro"] = float(f1_score(true, pred, average="macro", zero_division=0))



def train_one(args, train_df, internal_df, external_df, out_dir, garden_label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tf = make_transform(args.img_size, train=True, photometric_aug=args.photometric_aug, random_hflip=False)
    eval_tf = make_transform(args.img_size, train=False)

    train_dataset = PatientXrayDataset(
        train_df,
        train_tf,
        garden_label_map,
        image_col=args.image_col,
        max_views=args.max_views,
        train=True,
        shared_hflip_p=0.5,
    )
    internal_dataset = PatientXrayDataset(
        internal_df,
        eval_tf,
        garden_label_map,
        image_col=args.image_col,
        max_views=args.max_views,
        train=False,
    )
    external_dataset = PatientXrayDataset(
        external_df,
        eval_tf,
        garden_label_map,
        image_col=args.image_col,
        max_views=args.max_views,
        train=False,
    )
    
    print(
        "病人级样本数："
        f"train={len(train_dataset)} (双视图={int(train_dataset.df['has_lateral'].sum())}), "
        f"internal={len(internal_dataset)} (双视图={int(internal_dataset.df['has_lateral'].sum())}), "
        f"external={len(external_dataset)} (双视图={int(external_dataset.df['has_lateral'].sum())})"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    internal_loader = DataLoader(
        internal_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    external_loader = DataLoader(
        external_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    fracture_classes = infer_num_classes(train_df, internal_df, external_df, column="fracture", minimum=2)
    garden_classes = len(garden_label_map)
    garden_index_to_label = {idx: label for label, idx in garden_label_map.items()}
    
    model = MultiTaskModel(
        args.arch,
        pretrained=not args.no_pretrained,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        fracture_classes=fracture_classes,
        garden_classes=garden_classes,
        use_cross_attention=not args.no_cross_attention,
        cross_attn_dim=args.cross_attn_dim,
    ).to(device)
    
    reg_loss = nn.L1Loss() if args.reg_loss == "l1" else nn.SmoothL1Loss(beta=5.0)
    fracture_loss = nn.CrossEntropyLoss()
    garden_loss = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.cosine else None
    best = None
    best_state = None
    best_score = None
    history = []
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_fracture_loss = 0.0
        total_garden_loss = 0.0
        total_angle_loss = 0.0
        total_seen = 0
        total_garden_seen = 0
        total_angle_seen = 0
    
        for x_ap, x_lat, has_lateral, fracture_y, garden_y, angle_y, _, _ in train_loader:
            x_ap = x_ap.to(device)
            x_lat = x_lat.to(device)
            has_lateral = has_lateral.to(device)
            fracture_y = fracture_y.to(device)
            garden_y = garden_y.to(device)
            angle_y = angle_y.to(device)
    
            angle_pred, fracture_logits, garden_logits = model(x_ap, x_lat=x_lat, has_lateral=has_lateral)
    
            loss_fracture = fracture_loss(fracture_logits, fracture_y)
            loss = args.fracture_weight * loss_fracture
    
            # garden 和 angle 只在 fracture=1 的样本上计算；
            # 若标签缺失，则对应样本不会进入该分支损失。
            garden_mask = (fracture_y == 1) & (garden_y >= 0)
            angle_mask = (fracture_y == 1) & torch.isfinite(angle_y)
    
            if torch.any(garden_mask):
                loss_garden = garden_loss(garden_logits[garden_mask], garden_y[garden_mask])
                loss = loss + args.garden_weight * loss_garden
                total_garden_loss += float(loss_garden.item()) * int(garden_mask.sum().item())
                total_garden_seen += int(garden_mask.sum().item())
    
            if torch.any(angle_mask):
                loss_angle = reg_loss(angle_pred[angle_mask], angle_y[angle_mask])
                loss = loss + args.reg_weight * loss_angle
                total_angle_loss += float(loss_angle.item()) * int(angle_mask.sum().item())
                total_angle_seen += int(angle_mask.sum().item())
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_n = len(x_ap)
            total_loss += float(loss.item()) * batch_n
            total_fracture_loss += float(loss_fracture.item()) * batch_n
            total_seen += batch_n
    
        if scheduler is not None:
            scheduler.step()
    
        internal_metrics, _ = predict_and_score(
            model,
            internal_loader,
            device,
            garden_index_to_label=garden_index_to_label,
            tta_flip=args.tta_flip,
        )
        external_metrics, _ = predict_and_score(
            model,
            external_loader,
            device,
            garden_index_to_label=garden_index_to_label,
            tta_flip=args.tta_flip,
        )
    
        score = internal_metrics.get("angle_mae", float("nan"))
        if not np.isfinite(score):
            score = -internal_metrics.get("fracture_f1_macro", 0.0)
    
        if best_score is None or score < best_score:
            best_score = score
            best = {"epoch": epoch, "internal": internal_metrics, "external": external_metrics}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
        row = {
            "epoch": epoch,
            "loss": total_loss / max(total_seen, 1),
            "fracture_loss": total_fracture_loss / max(total_seen, 1),
            "garden_loss_pos_only": total_garden_loss / max(total_garden_seen, 1),
            "angle_loss_pos_only": total_angle_loss / max(total_angle_seen, 1),
            **{f"internal_{k}": v for k, v in internal_metrics.items()},
            **{f"external_{k}": v for k, v in external_metrics.items()},
        }
        history.append(row)
    
        print(
            f"epoch={epoch:02d} "
            f"loss={row['loss']:.4f} "
            f"internal_fracture_acc={internal_metrics['fracture_acc']:.3f} "
            f"internal_garden_acc={internal_metrics['garden_acc']:.3f} "
            f"internal_angle_mae={internal_metrics['angle_mae']:.3f} "
            f"external_fracture_acc={external_metrics['fracture_acc']:.3f} "
            f"external_garden_acc={external_metrics['garden_acc']:.3f} "
            f"external_angle_mae={external_metrics['angle_mae']:.3f}"
        )
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best": best,
            "garden_label_map": garden_label_map,
            "garden_index_to_label": garden_index_to_label,
            "fracture_classes": fracture_classes,
            "garden_classes": garden_classes,
        },
        out_dir / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    (out_dir / "best_summary.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, best, garden_index_to_label


@torch.no_grad()
def predict_and_score(model, loader, device, garden_index_to_label=None, tta_flip=False):
    model.eval()
    garden_index_to_label = garden_index_to_label or {}

    ids = []
    n_views_all = []
    angle_true = []
    angle_pred = []
    fracture_true = []
    fracture_pred = []
    fracture_probs = []
    garden_true_idx = []
    garden_pred_idx = []
    garden_probs = []
    
    for x_ap, x_lat, has_lateral, fracture_y, garden_y, angle_y, image_id, n_views in loader:
        x_ap = x_ap.to(device)
        x_lat = x_lat.to(device)
        has_lateral = has_lateral.to(device)
        angle_hat, fracture_logits, garden_logits = model(x_ap, x_lat=x_lat, has_lateral=has_lateral)
    
        if tta_flip:
            angle_hat_flip, fracture_logits_flip, garden_logits_flip = model(
                torch.flip(x_ap, dims=[3]),
                x_lat=torch.flip(x_lat, dims=[3]),
                has_lateral=has_lateral,
            )
            angle_hat = (angle_hat + angle_hat_flip) / 2
            fracture_logits = (fracture_logits + fracture_logits_flip) / 2
            garden_logits = (garden_logits + garden_logits_flip) / 2
    
        fracture_prob = torch.softmax(fracture_logits, dim=1)
        garden_prob = torch.softmax(garden_logits, dim=1)
    
        ids.extend(image_id.cpu().numpy().tolist())
        n_views_all.extend(n_views.cpu().numpy().tolist())
        angle_true.extend(angle_y.cpu().numpy().tolist())
        angle_pred.extend(angle_hat.cpu().numpy().tolist())
        fracture_true.extend(fracture_y.cpu().numpy().tolist())
        fracture_pred.extend(torch.argmax(fracture_logits, dim=1).cpu().numpy().tolist())
        fracture_probs.extend(fracture_prob.cpu().numpy().tolist())
        garden_true_idx.extend(garden_y.cpu().numpy().tolist())
        garden_pred_idx.extend(torch.argmax(garden_logits, dim=1).cpu().numpy().tolist())
        garden_probs.extend(garden_prob.cpu().numpy().tolist())
    
    angle_true = np.array(angle_true, dtype=float)
    angle_pred = np.array(angle_pred, dtype=float)
    fracture_true = np.array(fracture_true, dtype=int)
    fracture_pred = np.array(fracture_pred, dtype=int)
    fracture_probs = np.array(fracture_probs, dtype=float)
    garden_true_idx = np.array(garden_true_idx, dtype=int)
    garden_pred_idx = np.array(garden_pred_idx, dtype=int)
    garden_probs = np.array(garden_probs, dtype=float)
    n_views_all = np.array(n_views_all, dtype=int)
    
    metrics = {"n": int(len(fracture_true)), "n_two_view": int(np.sum(n_views_all >= 2))}
    add_classification_metrics(metrics, "fracture", fracture_true, fracture_pred)
    
    # 与训练保持一致：Garden 分类和 angle 回归只在真实 fracture=1 且标签存在的样本上评价。
    angle_mask = (fracture_true == 1) & np.isfinite(angle_true)
    add_regression_metrics(metrics, "angle", angle_true[angle_mask], angle_pred[angle_mask])
    
    garden_mask = (fracture_true == 1) & (garden_true_idx >= 0)
    add_classification_metrics(metrics, "garden", garden_true_idx[garden_mask], garden_pred_idx[garden_mask])
    
    def decode_garden(v):
        if int(v) < 0:
            return np.nan
        return garden_index_to_label.get(int(v), int(v))
    
    true_garden = [decode_garden(v) for v in garden_true_idx]
    pred_garden = [decode_garden(v) for v in garden_pred_idx]
    
    pred_data = {
        "id": ids,
        "n_views": n_views_all,
        "used_cross_attention": n_views_all >= 2,
        "true_fracture": fracture_true,
        "pred_fracture": fracture_pred,
    }
    
    # 输出 fracture 每个类别的预测概率。
    # 例如二分类时会新增：fracture_prob_0, fracture_prob_1。
    for class_idx in range(fracture_probs.shape[1]):
        pred_data[f"fracture_prob_{class_idx}"] = fracture_probs[:, class_idx]
    pred_data["pred_fracture_probability"] = fracture_probs.max(axis=1)
    
    pred_data.update(
        {
            "true_garden": true_garden,
            "pred_garden": pred_garden,
        }
    )
    
    # 输出 garden 每个类别的预测概率，列名使用原始 garden 标签。
    # 例如原始标签为 1/2/3/4 时会新增：garden_prob_1, garden_prob_2, garden_prob_3, garden_prob_4。
    for class_idx in range(garden_probs.shape[1]):
        raw_label = garden_index_to_label.get(class_idx, class_idx)
        pred_data[f"garden_prob_{raw_label}"] = garden_probs[:, class_idx]
    pred_data["pred_garden_probability"] = garden_probs.max(axis=1)
    
    pred_data.update(
        {
            "true_angle": angle_true,
            "pred_angle": angle_pred,
        }
    )
    
    preds = pd.DataFrame(pred_data)
    preds["angle_abs_error"] = (preds["pred_angle"] - preds["true_angle"]).abs()
    preds["angle_diff"] = preds["pred_angle"] - preds["true_angle"]
    preds["mean_angle"] = (preds["pred_angle"] + preds["true_angle"]) / 2
    preds.loc[preds["true_fracture"] != 1, ["angle_abs_error", "angle_diff", "mean_angle"]] = np.nan
    
    return metrics, preds

def save_error_summary(preds, split_name, out_dir):
    # Angle/Garden 只针对真实 fracture=1 且标签存在的样本做分层误差统计。
    valid_angle = preds[(preds["true_fracture"] == 1) & preds["true_angle"].notna()].copy()
    if len(valid_angle) > 0:
        if "true_garden" in valid_angle.columns:
            by_garden = valid_angle.groupby("true_garden", dropna=True)["angle_abs_error"].agg(["count", "mean", "median", "max"]).reset_index()
            by_garden.insert(0, "split", split_name)
            by_garden.to_csv(out_dir / f"{split_name}_angle_error_by_garden.csv", index=False, encoding="utf-8-sig")

        bins = [0, 30, 50, 90]
        labels = ["<30", "30-50", ">50"]
        valid_angle["angle_bin"] = pd.cut(valid_angle["true_angle"], bins=bins, labels=labels, include_lowest=True)
        by_angle = valid_angle.groupby("angle_bin", observed=True)["angle_abs_error"].agg(["count", "mean", "median", "max"]).reset_index()
        by_angle.insert(0, "split", split_name)
        by_angle.to_csv(out_dir / f"{split_name}_angle_error_by_angle_bin.csv", index=False, encoding="utf-8-sig")
    
    fracture_summary = preds.groupby(["true_fracture", "pred_fracture"]).size().reset_index(name="count")
    fracture_summary.insert(0, "split", split_name)
    fracture_summary.to_csv(out_dir / f"{split_name}_fracture_confusion_counts.csv", index=False, encoding="utf-8-sig")
    
    valid_garden = preds[(preds["true_fracture"] == 1) & preds["true_garden"].notna()].copy()
    if len(valid_garden) > 0:
        garden_summary = valid_garden.groupby(["true_garden", "pred_garden"]).size().reset_index(name="count")
        garden_summary.insert(0, "split", split_name)
        garden_summary.to_csv(out_dir / f"{split_name}_garden_confusion_counts.csv", index=False, encoding="utf-8-sig")



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--arch",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
            "efficientnet_v2_s",
            "convnext_tiny",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
            "shufflenet_v2_x1_5",
            "shufflenet_v2_x2_0",
        ],
        default="resnet34",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-loss", choices=["l1", "huber"], default="l1")
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--fracture-weight", type=float, default=0.5)
    parser.add_argument("--garden-weight", type=float, default=0.5)
    parser.add_argument("--cls-weight", type=float, default=None, help="兼容旧参数：若提供，则覆盖 --garden-weight。")
    parser.add_argument("--head-hidden", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--photometric-aug", action="store_true")
    parser.add_argument("--tta-flip", dest="tta_flip", action="store_true", default=True)
    parser.add_argument("--no-tta-flip", dest="tta_flip", action="store_false")
    parser.add_argument("--weights-dir", type=Path, default=Path("/home/panhuiqing/params"))
    parser.add_argument("--run-name", type=str, default="cross_attention")
    parser.add_argument("--image-col", type=str, default="path", help="CSV 中用于读取 ROI/图像文件的路径列，默认 path。")
    parser.add_argument("--max-views", type=int, default=2, help="每个 id 最多使用几张图；cross-attention 使用前两张。")
    parser.add_argument("--cross-attn-dim", type=int, default=0, help="Q/K/V 投影维度；0 表示使用 backbone 特征维度。")
    parser.add_argument("--no-cross-attention", action="store_true", help="关闭双视图 cross-attention，仅用第一张图。")
    args = parser.parse_args()

    if args.cls_weight is not None:
        args.garden_weight = args.cls_weight
    if args.max_views < 1:
        raise ValueError("--max-views 必须 >= 1")
    
    set_seed(args.seed)
    
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(args.weights_dir))
    
    print(f"预训练权重目录：{torch.hub.get_dir()}")
    print(
        "Cross-attention："
        f"{'关闭' if args.no_cross_attention else '开启'}；"
        f"图像路径列={args.image_col}；每个 id 最多使用 {args.max_views} 张图。"
    )
    
    out_dir = args.root / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = read_file(args.root / "data" / "train.csv", image_col=args.image_col)
    internal_test_df = read_file(args.root / "data" / "internal.csv", image_col=args.image_col)
    external = read_file(args.root / "data" / "external.csv", image_col=args.image_col)
    
    garden_label_map = build_label_map(train_df, internal_test_df, external, column="garden")
    print(f"Garden 标签映射：{garden_label_map}")
    print("说明：fracture 分类损失计算所有样本；garden 分类损失和 angle 回归损失只计算 fracture=1 且标签存在的样本。")
    
    model, best, garden_index_to_label = train_one(args, train_df, internal_test_df, external, out_dir, garden_label_map)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_tf = make_transform(args.img_size, train=False)
    model.to(device)
    
    all_metrics = {}
    for split_name, split_df in [("internal", internal_test_df), ("external", external)]:
        eval_dataset = PatientXrayDataset(
            split_df,
            eval_tf,
            garden_label_map,
            image_col=args.image_col,
            max_views=args.max_views,
            train=False,
        )
        loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        metrics, preds = predict_and_score(
            model,
            loader,
            device,
            garden_index_to_label=garden_index_to_label,
            tta_flip=args.tta_flip,
        )
    
        meta_cols = [
            c
            for c in ["id", "injury_side", "path", "view_paths", "source_image", "source_row", "raw_id", "n_views"]
            if c in eval_dataset.df.columns
        ]
        if meta_cols:
            meta = eval_dataset.df[meta_cols].copy()
            # n_views already exists in preds; keep the prediction-side value.
            duplicate_cols = [c for c in meta.columns if c in preds.columns and c != "id"]
            meta = meta.drop(columns=duplicate_cols)
            preds = preds.merge(meta, on="id", how="left")
    
        preds.to_csv(out_dir / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        save_error_summary(preds, split_name, out_dir)
        all_metrics[split_name] = metrics
        print(f"FINAL {split_name} {metrics}")
    
    pd.DataFrame(all_metrics).T.to_csv(out_dir / "metrics.csv", encoding="utf-8-sig")

if __name__ == "__main__":
    main()
