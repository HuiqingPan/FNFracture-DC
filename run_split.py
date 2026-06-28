

import pandas as pd
import os
from sklearn.model_selection import train_test_split

# data_cleaned = pd.read_excel("data/data_cleaned - 按图片重新评.xlsx", dtype={'fracture': str})   # 每个视频的 frames CSV 文件夹
# output_folder = "data/"       # 保存 train/val/test CSV
# os.makedirs(output_folder, exist_ok=True)
#
# train_data, temp_data = train_test_split(data_cleaned,
#     test_size=0.4,
#     stratify=data_cleaned['fracture'],  # 分层
#     random_state=0
# )
#
# val_data, internal_data = train_test_split(temp_data,
#     test_size=0.5,
#     stratify=temp_data["fracture"],  # 分层
#     random_state=0
# )
#
# # ==== 保存 CSV ====
# train_data.to_csv(os.path.join(output_folder, "train.csv"), index=False)
# val_data.to_csv(os.path.join(output_folder, "val.csv"), index=False)
# internal_data.to_csv(os.path.join(output_folder, "internal.csv"), index=False)
#
# print(f"划分完成，train: {len(train_data)}行, val: {len(val_data)}行, test: {len(internal_data)}行")











