import pandas as pd

class calculate_mapping:
    def calc_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - inter_area
        if union == 0:
            return 0.0
        return inter_area / union

    def calc_dice(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter_area = inter_w * inter_h

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        if area1 + area2 == 0:
            return 0.0
        return 2 * inter_area / (area1 + area2)

    # mapping rule设置
    def load_laterality(self, path, col_seq):
        """从excel文件加载影像侧别标注"""
        df = pd.read_excel(path, dtype={col_seq: str})
        # excel有两列：'image_name'（不含后缀）和 'image_laterality'
        laterality_dict = {}
        for _, row in df.iterrows():
            laterality_dict[row['image_name']] = row['image_laterality']
        print(laterality_dict)
        return laterality_dict

    def crop_with_padding(self, img, xyxy, pad_ratio):
        """带padding的裁剪"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = xyxy

        # 计算padding
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio

        # 计算新的边界（确保不超出图像范围）
        nx1 = max(0, int(x1 - px))
        ny1 = max(0, int(y1 - py))
        nx2 = min(w - 1, int(x2 + px))
        ny2 = min(h - 1, int(y2 + py))

        # 裁剪
        crop = img[ny1:ny2, nx1:nx2]

        return crop if crop.size > 0 else None

    def select_and_crop_by_image_laterality(self, img, boxes, laterality, img_width, pad_ratio=0.2):
        """
        根据影像侧别选择并裁剪对应的检测框

        参数:
            img: 原始图像 (numpy array)
            boxes: YOLO检测框对象
            laterality: 影像侧别 ('L'/'R)
            img_width: 图像宽度
            pad_ratio: 裁剪时的padding比例

        返回:
            crop: 裁剪后的图像区域 (None表示未找到合适的框)
            selected_box: 选择的边界框坐标 [x1, y1, x2, y2]
        """
        if boxes is None or len(boxes) == 0:
            return None, None

        # 计算图像中线
        midline = img_width / 2

        # 分离左右侧的检测框
        left_boxes = []
        right_boxes = []

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            conf = box.conf.cpu().numpy()[0]

            # 计算框的中心点x坐标
            center_x = (x1 + x2) / 2

            # 根据中心点位置分类
            if center_x < midline:
                left_boxes.append({
                    'index': i,
                    'xyxy': xyxy,
                    'conf': conf,
                    'center_x': center_x
                })
            else:
                right_boxes.append({
                    'index': i,
                    'xyxy': xyxy,
                    'conf': conf,
                    'center_x': center_x
                })

        # 根据影像侧别选择对应的框
        if laterality == '左':
            candidates = left_boxes
            print(f"  选择影像左侧框，找到 {len(left_boxes)} 个候选框")
        elif laterality == '右':
            candidates = right_boxes
            print(f"  选择影像右侧框，找到 {len(right_boxes)} 个候选框")
        else:
            # 如果laterality不是'left'或'right'，使用全局最高置信度
            print(f"  警告：未知侧别 '{laterality}'，使用全局最高置信度框")
            best_i = boxes.conf.cpu().numpy().argmax()
            xyxy = boxes.xyxy[best_i].cpu().numpy()
            return self.crop_with_padding(img, xyxy, pad_ratio), xyxy

        # 如果没有找到对应侧的框
        if not candidates:
            print(f"  警告：未在影像{laterality}侧检测到框，使用全局最高置信度框")
            best_i = boxes.conf.cpu().numpy().argmax()
            xyxy = boxes.xyxy[best_i].cpu().numpy()
            return self.crop_with_padding(img, xyxy, pad_ratio), xyxy

        # 在对应侧的框中选置信度最高的
        best_candidate = max(candidates, key=lambda x: x['conf'])
        xyxy = best_candidate['xyxy']

        print(f"  选择置信度 {best_candidate['conf']:.3f} 的框 (中心x={best_candidate['center_x']:.0f})")

        return self.crop_with_padding(img, xyxy, pad_ratio), xyxy