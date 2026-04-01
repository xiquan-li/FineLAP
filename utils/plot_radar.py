import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from typing import List, Optional
import warnings

# 检测并设置字体
def setup_font():
    """检测并设置Times New Roman字体，如果不存在则使用备用字体"""
    # 尝试的字体列表（按优先级）
    font_candidates = [
        'Times New Roman',
        'Times',
        'DejaVu Serif',
        'Liberation Serif',
        'serif'
    ]
    
    # 获取所有可用字体
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
    # 查找第一个可用的字体
    selected_font = None
    for font in font_candidates:
        if font in available_fonts or font == 'serif':
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.family'] = selected_font
        if selected_font != 'serif':
            plt.rcParams['font.serif'] = [selected_font]
        print(f"使用字体: {selected_font}")
        return selected_font
    else:
        # 如果都找不到，使用默认serif字体
        plt.rcParams['font.family'] = 'serif'
        warnings.warn("未找到Times New Roman，使用默认serif字体")
        return 'serif'

# 设置字体
FONT_NAME = setup_font()


def radar_plot(
    categories: List[str],
    values: List[List[float]],  # 多个系列的数据，每个系列是一个列表
    labels: List[str],  # 每个系列的标签（模型名称）
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,  # 线型：'-', '--', '-.', ':'
    markers: Optional[List[str]] = None,  # 标记：'o', 's', '^', 'v', 'D'
    fill_alpha: float = 0.15,
    line_width: float = 2.0,
    min_value: float = 0.0,
    save_path: str = "radar_plot.png",
    dpi: int = 300,
    figsize: tuple = (10, 10)
):
    """
    绘制雷达图（支持多模型对比，每个轴独立比例）
    
    Args:
        categories: 类别名称列表（轴标签）
        values: 多个系列的数据，每个系列是一个数值列表
        labels: 每个系列的标签（用于图例）
        colors: 颜色列表
        linestyles: 线型列表（'-', '--', '-.', ':'）
        markers: 标记列表（'o', 's', '^', 'v', 'D'）
        fill_alpha: 填充透明度
        line_width: 线宽
        min_value: 最小值
        save_path: 保存路径
        dpi: 分辨率
        figsize: 图片大小
    """
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    category_max_values = []
    for i in range(num_vars):
        category_values = [v[i] for v in values]
        max_val = max(category_values)
        MERGIN = 1.15
        if max_val < 100:
            category_max = max_val * MERGIN
        else:
            category_max = max(max_val * MERGIN, 100)
        category_max_values.append(category_max)
    
    normalized_values = []
    for value_list in values:
        normalized = []
        for i, val in enumerate(value_list):
            normalized_val = (val / category_max_values[i]) * 100
            normalized.append(normalized_val)
        normalized_values.append(normalized)
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_rticks([])
    
    grid_values = np.linspace(0, 100, 5)
    for value in grid_values:
        ax.plot(angles, [value] * len(angles), 'k-', alpha=0.2, linewidth=0.5, linestyle='--')
    
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], 'k-', alpha=0.1, linewidth=0.3)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([''] * len(categories))
    
    radius_add = [5, 10, 17.5, 17.5, 10, 5, 10, 12.5, 15, 7.5]
    label_radius = 100
    for angle, category, r in zip(angles[:-1], categories, radius_add):
        ax.text(angle, label_radius + r, category, 
               horizontalalignment='center', verticalalignment='center',
               fontsize=18, fontweight='bold', fontfamily=FONT_NAME)
    
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', '-', '--']
    if markers is None:
        markers = ['s', 's', 'o', 'o', '^', 'o']
    
    for idx, value_list in enumerate(values):
        normalized_list = normalized_values[idx] + normalized_values[idx][:1]
        original_list = value_list + value_list[:1]
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        
        ax.plot(angles, normalized_list, linestyle=linestyle, linewidth=line_width,
                marker=marker, markersize=4, color=color, label=labels[idx])
        
        if fill_alpha > 0:
            ax.fill(angles, normalized_list, alpha=fill_alpha, color=color)
        
        if idx == 0:
            radius_add = [2, 4, 8, 8, 8, 7, 8, 8, 7, 4]
            for angle, norm_val, orig_val, ra in zip(angles[:-1], normalized_list[:-1], original_list[:-1], radius_add):
                ax.text(angle, norm_val + ra, f'{orig_val}',
                       horizontalalignment='center', verticalalignment='bottom',
                       fontsize=12, color=color, fontweight='bold', fontfamily=FONT_NAME)
    
    legend = ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.05), 
                       framealpha=0.9, prop={'family': FONT_NAME, 'size': 12})
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    
    return fig, ax


# ========== Demo 示例 ==========
if __name__ == "__main__":
    # 示例：多模型对比（类似图片中的效果）

    # ================================ 带VGGSound ================================
    # categories = ['AudioCaps A2T', 'AudioCaps T2A', 'Clotho A2T', 'Clotho T2A', 
    #             'DESED', 'AudioSet-SL', 'UrbanSED', 'TAG', 
    #             'ESC-50', 'US8K', 'VGGSound']
    
    # values = [
    #     [45.8, 62.6, 19.3, 26.5, 0.344, 0.471, 0.446, 0.608, 92.1, 84.9, 31.2],  # FineLAP
    #     [42.2, 53.7, 20.8, 26.5, 0.264, 0.354, 0.087, 0.487, 94.9, 83.7, 31.8],  # MGA-CLAP
    #     [32.1, 43.3, 13.8, 16.7, 0.094, 0.351, 0.295, 0.455, 86.9, 75.6, 39.3],  # FLAM
    #     [39.7, 51.7, 19.5, 23.4, 0.131, 0.284, 0.016, 0.344, 94.8, 80.6, 29.6],  # HTSAT-BERT
    # ]
    #  ================================ 带VGGSound ================================


    # ================================ 不带VGGSound ================================
    categories = ['AudioCaps A2T', 'AudioCaps T2A', 'Clotho A2T', 'Clotho T2A', 
                'DESED', 'AudioSet-SL', 'UrbanSED', 'TAG', 
                'ESC-50', 'US8K']
    
    values = [
        [45.7, 62.5, 18.9, 26.6, 0.35, 0.47, 0.45, 0.65, 93.9, 84.9],  # FineLAP
        [42.2, 53.7, 20.8, 26.5, 0.264, 0.354, 0.087, 0.487, 94.9, 83.7],  # MGA-CLAP
        [32.1, 43.3, 13.8, 16.7, 0.094, 0.351, 0.295, 0.455, 86.9, 75.6],  # FLAM
        [39.7, 51.7, 19.5, 23.4, 0.131, 0.284, 0.016, 0.344, 94.8, 80.6],  # HTSAT-BERT
    ]
    # ================================ 不带VGGSound ================================


    labels = ['FineLAP (Ours)', 'MGA-CLAP', 'FLAM', 'HTSAT-BERT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#808080']
    linestyles = ['-', '--', '-.', '--']
    markers = ['s', '^', 'o', 'o']
    
    fig, ax = radar_plot(
        categories=categories,
        values=values,
        labels=labels,
        colors=colors,
        linestyles=linestyles,
        markers=markers,
        fill_alpha=0.1,
        line_width=2.0,
        save_path="/inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/test/radar_performance.png"
    )
    
    plt.show()
