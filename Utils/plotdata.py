# -*- coding: utf-8 -*-
"""
Facet grouped bar + point (journal-friendly)
三列分面：2.4 / 5 / 28 GHz
每列 X 轴为房间（1..4），每个房间里三根细柱：Tx1 / Tx12 / Tx123，并在柱顶叠加散点
配色：Okabe–Ito（色盲安全）
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ============== 路径与画布 ==============
OUTPUT_DIR = r"E:\manuscript\journal\2025Tap\paper\img"  # ← 改成你的路径
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGSIZE = (10.2, 3.6)  # 横向排三列
DPI = 600

# 期刊字体
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 颜色（Okabe–Ito）
COL_TX1   = "#0072B2"  # 蓝
COL_TX12  = "#D55E00"  # 橙红
COL_TX123 = "#009E73"  # 蓝绿

# 柱宽与间距
BAR_W = 0.22
EDGE_LW = 0.7
EDGE_COL = "black"
BAR_ALPHA = 0.85

MARKER_SIZE = 26
MARKER_EDGE = "white"
MARKER_EW   = 0.7

# ============== 数据 ==============
tx1 = np.array([
    [2.4,1,-0.0177726562499991],[2.4,2,0.108244267956998],[2.4,3,-0.313624104034901],[2.4,4,0.25665],
    [5,1,-0.0164819300000048],[5,2,0.975565790000005],[5,3,1.24315],[5,4,-1.3808756],
    [28,1,-0.309114697265599],[28,2,2.1652722310259],[28,3,1.65605],[28,4,-1.9467490886614]
], float)
tx12 = np.array([
    [2.4,1,0.1165],[2.4,2,0.2905],[2.4,3,0.34557],[2.4,4,0.84769],
    [5,1,0.1725],[5,2,0.20719],[5,3,0.6071],[5,4,1.26452],
    [28,1,0.09297],[28,2,0.07002],[28,3,1.92875],[28,4,0.77515]
], float)
tx123 = np.array([
    [2.4,1,-0.143],[2.4,2,0.0666],[2.4,3,0.13376],[2.4,4,-1.0908],
    [5,1,-0.25085],[5,2,-0.25605],[5,3,0.4833],[5,4,-0.89685],
    [28,1,0.10183],[28,2,0.08008],[28,3,-0.27116],[28,4,-0.52211]
], float)

# 取绝对值（你的需求是绝对误差）
for arr in (tx1, tx12, tx123):
    arr[:,2] = np.abs(arr[:,2])

ROOM_IDS   = np.array([1,2,3,4])
FREQ_LABELS = [2.4, 5.0, 28.0]
SERIES = [
    ("Tx in Room 1",          tx1,   COL_TX1),
    ("Tx in Rooms 1 and 2",   tx12,  COL_TX12),
    ("Tx in Rooms 1, 2 and 3",tx123, COL_TX123),
]

# 计算统一的 y 轴上限，便于跨分面比较
z_max = max(tx1[:,2].max(), tx12[:,2].max(), tx123[:,2].max())
step  = 0.5 if z_max <= 3 else 1.0
ylim_top = float(np.ceil(z_max/step)*step)

# ============== 绘图 ==============
fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True, sharey=True)

# 每个分面：一个频点
for ax, f in zip(axes, FREQ_LABELS):
    # 三条序列在一个房间中的偏移（-w, 0, +w）
    offsets = [-BAR_W, 0.0, BAR_W]
    for (label, data, color), dx in zip(SERIES, offsets):
        sel = data[data[:,0] == f][:, 1:3]       # [room, value]
        order = np.argsort(sel[:,0])
        rooms = sel[order, 0].astype(int)
        vals  = sel[order, 1]

        pos = rooms + dx
        bars = ax.bar(pos, vals, width=BAR_W*0.95,
                      color=color, edgecolor=EDGE_COL, linewidth=EDGE_LW,
                      alpha=BAR_ALPHA, label=label)
        ax.scatter(pos, vals, s=MARKER_SIZE, c=color,
                   edgecolors=MARKER_EDGE, linewidths=MARKER_EW, zorder=3)

    ax.set_title(f"{f:g} GHz", fontsize=12, pad=8)
    ax.set_xticks(ROOM_IDS)
    ax.set_xlabel("Room ID", fontsize=11)
    ax.set_xlim(0.5, 4.5)
    ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.5)
    # 去掉上右脊
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

axes[0].set_ylabel("Absolute Error (dB)", fontsize=11)
axes[0].set_ylim(0.0, ylim_top)
axes[0].set_yticks(np.arange(0.0, ylim_top + 1e-9, step))

# 统一图例（上方居中）
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, ncol=3, loc='upper center', frameon=True, fontsize=10, bbox_to_anchor=(0.5, 1.04))
leg.get_frame().set_alpha(0.95)

# 主标题（可选）
# fig.suptitle("Four-room Structure — Absolute Error by Frequency", fontsize=14, y=1.08)

# 导出
png_path = os.path.join(OUTPUT_DIR, "Four-room-Structure_faceted_bars_points.png")
pdf_path = os.path.join(OUTPUT_DIR, "Four-room-Structure_faceted_bars_points.pdf")
plt.savefig(png_path, dpi=DPI, bbox_inches='tight')
plt.savefig(pdf_path, dpi=DPI)  # 矢量，适合期刊
plt.close(fig)
print("Saved:", png_path)
print("Saved:", pdf_path)
