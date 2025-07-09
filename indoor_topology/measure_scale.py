import cv2
import matplotlib.pyplot as plt
import numpy as np
import json

img_path = r'E:\code\floor_data\cubicasa5k\high_quality_architectural\7795\svgImg_roughcast.png'  # 修改为你的文件名
real_len = 6  # 若想脚本里直接指定真实长度，可赋值；否则运行中手动输入

# 读取与显示
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("依次点击参考边的两个端点（点击完关闭窗口）")

# 记录两次点击
points = plt.ginput(2, timeout=0)  # 无限等待点击
plt.close(fig)

if len(points) != 2:
    raise RuntimeError("未检测到两次点击，请重试。")

# 计算像素距离
(x1, y1), (x2, y2) = points
L_px = np.hypot(x2 - x1, y2 - y1)

# 若未预设真实长度，要求用户输入
if real_len is None:
    real_len = float(input(f"参考边像素长度为 {L_px:.2f} px。\n请输入该边真实长度: "))

m_per_px = real_len / L_px
m2_per_px2 = m_per_px ** 2

print("\n--- 计算结果 ---")
print(f"像素长度  L_px        = {L_px:.2f} px")
print(f"真实长度  real_len    = {real_len:.4f} (同输入单位)")
print(f"长度系数  m_per_px    = {m_per_px:.6f} (真实单位 / px)")
print(f"面积系数  m2_per_px2  = {m2_per_px2:.6f} (真实单位² / px²)")

# # 可选：把系数保存到 JSON 以便后续读取
# with open('scale_factor.json', 'w', encoding='utf-8') as f:
#     json.dump({'m_per_px': m_per_px, 'm2_per_px2': m2_per_px2}, f, indent=2)
