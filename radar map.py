import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import math
import os

# =================================================================
# 绘图参数都
# =================================================================
STYLE_CONFIG = {
    "input_path": " ",  # 设置输入的Excel文件路径
    "sheet_name": "",
    "output_name": " ",  # 设置导出的图片文件名称
    "dpi": 300,  # 设置图片分辨率，600DPI达到印刷级清晰度

    "font_family_en": "sans-serif",  # 设置英文名
    # "font_family_zh": "SimSun",  # 设置中文名字体为宋体（SimSun）
    "label_size": 16,  # 设置雷达图最外圈模型名称的文字大小
    "tick_size": 14,  # 设置内部径向坐标轴数值的文字大小
    "legend_size": 16,  # 设置底部图例文字的大小

    # 定义折线的颜色序列，绘图时会循环从中取色
    "line_colors": ["#da8d85", "#39972d"],
    # 定义折线的标记点形状序列（如圆点、方块、三角等）
    # "line_markers": ["o", "s", "^", "D", "v", "p"],
    "line_markers": ["o", "s"],
    "line_width": 1.5,  # 设置折线的线条宽度
    "marker_size": 6,  # 设置折线上标记点的大小
    "fill_alpha": 0.05,  # 设置折线围成区域的填充透明度（0.05表示极浅）

    # 设置最外层装饰圆环的颜色序列，将按模型逐个循环使用
    "ring_colors": ['#e29d95', '#F5B2AE', '#F5C6C3', '#F8D6D4', '#F5E3E1', '#A2B7A8', '#b4cfb3', '#B7D4B9', '#CBE3CD', '#D4E6D2', '#D7E7D7'],
    "ring_thickness": 0.12,  # 设置彩色圆环的厚度（占径向长度的比例）
    "ring_alpha": 0.8,  # 设置彩色圆环的透明度

    "grid_color": "#7f7f7f",  # 设置网格线（射线和圆圈）的颜色
    "grid_style": "--",  # 设置网格线的样式为虚线
    "grid_width": 0.8,  # 设置网格线的线条宽度

    "data_range": (0.0, 1.0),  # 设置数据归一化后的区间，防止点落在中心或贴边
    "r_ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 设置径向轴上显示的刻度位置
    "r_tick_labels": ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],  # 设置刻度对应显示的文字内容

    "legend_cols": 4,  # 设置图例显示时的列数
}

# 设置全局字体参数，合并中英文字体列表
# plt.rcParams['font.sans-serif'] = [STYLE_CONFIG["font_family_en"], STYLE_CONFIG["font_family_zh"]]
plt.rcParams['font.sans-serif'] = [STYLE_CONFIG["font_family_en"]]
# 设置正常显示坐标轴上的负号，防止显示为方块
plt.rcParams['axes.unicode_minus'] = False


# =================================================================
# 核心逻辑函数
# =================================================================
def create_professional_radar():
    """创建专业雷达图的主函数"""

    # --- A. 数据准备 ---
    # 检查指定的Excel文件是否存在，如果不存在则自动生成随机数据
    if not os.path.exists(STYLE_CONFIG["input_path"]):  # 判断文件路径是否存在
        data = {'Model': [f'M{i + 1}' for i in range(12)]}  # 创建包含12个模型名称的字典
        for m in ['R2', 'RMSE', 'MAE', 'Overfit', 'Val_R2', 'Val_RMSE', 'Time', 'Complexity']:  # 遍历指标名
            data[m] = np.random.rand(12)  # 为每个指标生成12个0-1之间的随机数
        pd.DataFrame(data).to_excel(STYLE_CONFIG["input_path"], index=False)  # 将字典转换为DataFrame并保存到Excel

    # 使用pandas读取Excel文件数据
    df = pd.read_excel(STYLE_CONFIG["input_path"], sheet_name=STYLE_CONFIG["sheet_name"])  # 读取Excel内容到数据帧
    model_names = df.iloc[:, 0].astype(str).tolist()  # 获取第一列数据作为模型名称列表
    metric_names = df.columns[1:].tolist()  # 获取除第一列以外的所有列名作为指标列表
    num_models = len(model_names)  # 计算模型的总数量（决定雷达图的轴数）
    num_metrics = len(metric_names)  # 计算评价指标的总数量（决定绘制的线数）

    # --- B. 数据归一化 ---
    # 将不同量纲的数据映射到统一的数值区间 [0.2, 0.9]
    # df_norm = df.copy()  # 复制原始数据用于存放归一化后的值
    # r_min, r_max = STYLE_CONFIG["data_range"]  # 从配置中读取目标区间的最小值和最大值
    # for col in metric_names:  # 遍历每一个指标列进行处理
    #     v_min, v_max = df[col].min(), df[col].max()  # 计算当前列的原始最小值和最大值
    #     if v_max != v_min:  # 如果最大最小值不相等，则执行线性映射
    #         # 线性归一化公式：目标 = r_min + (原始 - 原最小) / (原最大 - 原最小) * (目标区间长度)
    #         df_norm[col] = r_min + (df[col] - v_min) / (v_max - v_min) * (r_max - r_min)
    #     else:  # 如果该列数据全部相同
    #         df_norm[col] = r_min  # 则统一赋予区间最小值

    # --- B. 数据归一化（修改为全局归一化）---
    df_norm = df.copy()
    r_min, r_max = STYLE_CONFIG["data_range"]

    # 获取所有需要归一化的数据（所有指标列的所有值）
    all_values = df[metric_names].values.flatten()  # 将所有指标数据展平为一维数组
    v_min_global = all_values.min()  # 全局最小值
    v_max_global = all_values.max()  # 全局最大值

    # 如果全局最大值不等于最小值，则进行全局线性映射
    if v_max_global != v_min_global:
        # 对所有指标列应用相同的全局归一化
        for col in metric_names:
            # 使用全局的 min/max 进行线性映射
            df_norm[col] = r_min + (df[col] - v_min_global) / (v_max_global - v_min_global) * (r_max - r_min)
    else:
        # 如果所有数据都相同，则全部赋值为 r_min
        for col in metric_names:
            df_norm[col] = r_min

    # --- C. 计算角度 ---
    # 在极坐标系中，将2π（360度）按照模型数量等分
    angles = np.linspace(0, 2 * np.pi, num_models, endpoint=False).tolist()  # 计算每个轴的弧度位置
    # 为了使多边形闭合，需要将起始角度再次添加到列表末尾
    angles_closed = angles + angles[:1]  # 首尾相连的角度序列

    # --- D. 创建画布 ---
    # 初始化一个极坐标系的图表对象
    fig, ax = plt.subplots(figsize=(10, 11), subplot_kw=dict(polar=True))  # 设置画布大小并指定为极坐标
    ax.set_theta_offset(np.pi / 2)  # 设置雷达图的起始角度在正上方（90度位置）
    ax.set_theta_direction(-1)  # 设置坐标轴顺时针方向增加

    # 使用cycle函数创建颜色和标记的无限循环迭代器
    c_iter = cycle(STYLE_CONFIG["line_colors"])  # 折线颜色迭代器
    m_iter = cycle(STYLE_CONFIG["line_markers"])  # 形状标记迭代器

    # --- E. 绘制指标折线 ---
    for metric in metric_names:  # 遍历每一个评价指标进行绘图
        # 获取当前指标在所有模型下的归一化数值，并补充首位数据以实现闭合
        values = df_norm[metric].tolist() + [df_norm[metric].iloc[0]]  # 闭合的数据序列
        c, m = next(c_iter), next(m_iter)  # 从迭代器中取出本轮使用的颜色和形状标记
        # 在极坐标系中绘制折线
        ax.plot(angles_closed, values, color=c, marker=m, label=metric,
                linewidth=STYLE_CONFIG["line_width"], markersize=STYLE_CONFIG["marker_size"])  # 绘制线条和点
        # 填充折线内部区域，增加视觉可辨识度
        ax.fill(angles_closed, values, color=c, alpha=STYLE_CONFIG["fill_alpha"])  # 填充透明阴影区域

    # --- F. 绘制外层彩色装饰环 (修改：相邻分区颜色不同且循环使用) ---
    ring_bottom, ring_w = 1.0, STYLE_CONFIG["ring_thickness"]  # 定义圆环的起始半径（1.0）和厚度
    sector_unit = (2 * np.pi) / num_models  # 计算单个模型所占的扇区弧度大小
    ring_c_iter = cycle(STYLE_CONFIG["ring_colors"])  # 创建外圈颜色的无限循环迭代器

    for i in range(num_models):  # 遍历每一个模型独立绘制其对应的圆环部分
        ang = angles[i]  # 获取当前模型轴线的中心角度
        c_ring = next(ring_c_iter)  # 从迭代器中获取下一个颜色，确保相邻颜色不同
        # 在极坐标中使用条形图函数 bar 绘制单个扇区色块
        ax.bar(ang, ring_w, width=sector_unit, bottom=ring_bottom,
               color=c_ring, edgecolor='white', alpha=STYLE_CONFIG["ring_alpha"], zorder=0)  # 绘制色块

        # 计算模型名称文字的旋转角度
        rot = -np.rad2deg(ang)  # 将弧度转为角度，用于文字旋转偏移
        # 如果文字处于圆环下方（90度到270度之间），旋转180度避免文字倒挂
        if 90 < np.rad2deg(ang) < 270: rot += 180
        # 在色块正中位置书写模型名称
        ax.text(ang, ring_bottom + ring_w / 2, model_names[i], rotation=rot,
                ha='center', va='center', fontsize=STYLE_CONFIG["label_size"], fontweight='bold')  # 写入文字

    # --- G. 坐标轴与网格定制 ---
    ax.set_xticks(angles)  # 设置极坐标轴射线的刻度位置
    ax.set_xticklabels([])  # 清空默认的X轴刻度文字

    # 禁用系统默认的径向网格线，改为手动绘制以控制长度
    ax.xaxis.grid(False)  # 隐藏默认射线网格
    # 手动绘制从中心到圆环内边缘（1.0位置）的虚线射线
    for ang in angles:  # 遍历每个模型的角度
        ax.plot([ang, ang], [0, ring_bottom],
                color=STYLE_CONFIG["grid_color"],
                linestyle=STYLE_CONFIG["grid_style"],
                linewidth=STYLE_CONFIG["grid_width"],
                zorder=1)  # 绘制自定义网格射线

    # 启用并定制圆圈样式的网格线（Y轴网格）
    ax.yaxis.grid(True, color=STYLE_CONFIG["grid_color"], linestyle=STYLE_CONFIG["grid_style"],
                  linewidth=STYLE_CONFIG["grid_width"])

    # 彻底禁用系统默认生成的 Y 轴（径向轴）刻度标签
    ax.set_yticklabels([])  # 清空系统标签
    ax.set_yticks(STYLE_CONFIG["r_ticks"])  # 设置手动指定的刻度位置

    # 手动在指定的角度（0度，即正上方）绘制径向轴的数值标签
    label_angle = 0  # 标签绘制的方位角
    for r, txt in zip(STYLE_CONFIG["r_ticks"], STYLE_CONFIG["r_tick_labels"]):  # 遍历刻度位置和对应文字
        # 在图上书写数值文字，并进行精细的位置偏移处理
        ax.text(label_angle, r - 0.01, txt,
                color="black",  # 设置文字颜色为黑色
                size=STYLE_CONFIG["tick_size"],  # 设置文字大小
                fontweight='bold',  # 字体加粗
                ha="center",  # 水平居中对齐
                va="top",  # 垂直方向顶部对齐
                zorder=10)  # 确保刻度标签位于所有图形元素最上层

    # 设置径向显示的总范围：从中心0到彩色圆环的最外沿
    ax.set_ylim(0, ring_bottom + ring_w)  # 设置半径限值

    # --- H. 底部图例设置 ---
    # 动态计算图例显示的列数
    actual_cols = min(num_metrics, STYLE_CONFIG["legend_cols"])  # 计算实际列数
    ax.legend(
        loc='upper center',  # 将图例定位锚点设在上方中心
        bbox_to_anchor=(0.0, -0.12, 1.0, 0.1),  # 将图例放置在绘图区下方的空白处
        mode="expand",  # 使图例水平铺满宽度
        ncol=actual_cols,  # 设置图例的列数
        borderaxespad=0,  # 设置边框与轴的间距
        frameon=False,  # 不显示图例的外边框
        fontsize=STYLE_CONFIG["legend_size"]  # 设置图例的字体大小
    )  # 渲染图例

    # --- I. 保存与展示 ---
    # 将最终图像保存到磁盘，使用紧凑布局
    plt.savefig(STYLE_CONFIG["output_name"], dpi=STYLE_CONFIG["dpi"], bbox_inches='tight')  # 保存文件
    print(f"成功导出图片：{STYLE_CONFIG['output_name']}")  # 打印控制台消息
    plt.show()  # 展示结果


# 脚本程序的执行入口
if __name__ == "__main__":  # 判断是否直接运行此脚本
    create_professional_radar()  # 执行创建雷达图的函数