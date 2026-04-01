import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings

def setup_font():
    font_candidates = [
        'Times New Roman',
        'Times',
        'DejaVu Serif',
        'Liberation Serif',
        'serif'
    ]
    
    available_fonts = [f.name for f in font_manager.fontManager.ttflist]
    
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
        plt.rcParams['font.family'] = 'serif'
        warnings.warn("未找到Times New Roman，使用默认serif字体")
        return 'serif'

FONT_NAME = setup_font()

data_scales = [0, 50_000, 100_000, 200_000, 500_000, 2_000_000]
data_scale_labels = ['0', '50K', '100K', '200K', '500K', '2.2M']
max_scale = max(data_scales)
data_scales_percent = [x / max_scale * 100 for x in data_scales]
# percent_labels = [f'{p:.1f}%' for p in data_scales_percent]
percent_labels = ['0%', '2.5%', '5%', '10%', '25%', '100%']

desed_psds1 = [0.3087, 0.3048, 0.3042, 0.3123, 0.3171, 0.3449]
as_sl_psds1 = [0.3900, 0.4145, 0.4293, 0.4445, 0.4587, 0.4710]
urbansed_psds1 = [0.2626, 0.2936, 0.3186, 0.3526, 0.3856, 0.4463]

# 配置选项：True = 只画 AudioSet-SL, False = 画两个图（AudioSet-SL 和 UrbanSED）
PLOT_AS_SL_ONLY = True

if PLOT_AS_SL_ONLY:
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 4.5))
    ax2 = None
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# 左图：AudioSet-SL
ax1.plot(
    data_scales_percent,
    as_sl_psds1,
    marker='s',
    linewidth=2,
    label='Performance on AudioSet-SL',
    color='#DAA520',
)

xytexts = [(-4, 8), (-8, 8), (-8, 8), (-8, 8), (-4, 6), (-0, 4)]
for x, y, label, xytext in zip(data_scales_percent, as_sl_psds1, data_scale_labels, xytexts):
    ax1.annotate(
        label,
        (x, y),
        textcoords="offset points",
        xytext=xytext, 
        ha='center',
        fontsize=12,
        alpha=0.85,
        fontfamily=FONT_NAME
    )

ax1.axhline(y=0.465, color='#333333', linestyle='--', linewidth=1.5)
ax1.text(-4, 0.466, 'SOTA on AudioSet-SL', fontsize=14, 
         verticalalignment='bottom', horizontalalignment='left',
         fontfamily=FONT_NAME, color='#333333')

ax1.set_xticks(data_scales_percent)
ax1.set_xticklabels(percent_labels, fontfamily=FONT_NAME, rotation=45, fontsize=10)
ax1.set_xlabel('Number of audio-caption data without-level annotations', fontfamily=FONT_NAME, fontweight='bold', fontsize=14)
ax1.set_ylabel('PSDS1', fontfamily=FONT_NAME, fontweight='bold', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(frameon=True, prop={'family': FONT_NAME, 'size': 16}, loc='lower right')

# 右图：UrbanSED
if not PLOT_AS_SL_ONLY:
    ax2.plot(
        data_scales_percent,
        urbansed_psds1,
        marker='^',
        linewidth=2,
        label='Performance on UrbanSED',
        color='#228B22'
    )

    xytexts = [(-4, 8), (-8, 8), (-8, 8), (-8, 8), (-4, 6), (-0, 4)]
    for x, y, label, xytext in zip(data_scales_percent, urbansed_psds1, data_scale_labels, xytexts):
        ax2.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=xytext, 
            ha='center',
            fontsize=12,
            alpha=0.85,
            fontfamily=FONT_NAME
        )

    ax2.set_xticks(data_scales_percent)
    ax2.set_xticklabels(percent_labels, fontfamily=FONT_NAME, rotation=45, fontsize=8)
    ax2.set_xlabel('Number of audio-caption data without frame-level annotations', fontfamily=FONT_NAME, fontweight='bold', fontsize=12)
    ax2.set_ylabel('PSDS1', fontfamily=FONT_NAME, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, prop={'family': FONT_NAME, 'size': 16}, loc='lower right')

plt.tight_layout()
plt.savefig(
    '/inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/test/data_scale_vs_frame_performance.png',
    dpi=300
)
plt.show()
print(f"File saved to: /inspire/hdd/project/embodied-multimodality/public/xqli/ALE/F-CLAP/test/data_scale_vs_frame_performance.png")
