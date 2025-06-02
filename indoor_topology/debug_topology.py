import matplotlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from indoor_topology.detect_adjacency import detect_adjacency
from indoor_topology.extract_rooms import extract_rooms
from indoor_topology.save_topology_image import save_topology_image, save_to_excel

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def main():
    # ---------- 路径 ----------
    wall_svg_path = "wall_svg.png"
    icon_svg_path = "icon_svg.png"
    rough_img_path = "svgImg_roughcast.png"  # 作为背景
    palette_img_path = "icon.png"

    # ---------- 加载墙体索引图 ----------
    wall_label_img = Image.open(wall_svg_path).convert('P')
    wall_label_array = np.array(wall_label_img)

    # ---------- 提取房间 ----------
    region_id_map, rooms, min_len = extract_rooms(wall_label_array)

    # ---------- 生成墙 / 图标数组 ----------
    wall_array = (wall_label_array == 2).astype(np.uint8)
    icon_label_img = Image.open(icon_svg_path).convert('P')
    icon_label_array = np.array(icon_label_img)
    icon_array = np.zeros_like(icon_label_array, dtype=np.uint8)
    icon_array[icon_label_array == 1] = 1  # 门
    icon_array[icon_label_array == 2] = 2  # 窗

    # ---------- 邻接检测 ----------
    edges = detect_adjacency(region_id_map, wall_array, icon_array,
                             wall_label_array, max_wall_thickness=min_len)

    # ---------- 保存拓扑图 & Excel ----------
    base = os.path.splitext(os.path.basename(wall_svg_path))[0]
    img_save = f"debug_topology_result.png"
    xlsx_save = f"debug_topology_edges.xlsx"

    save_topology_image(region_id_map, rooms, edges, rough_img_path, img_save)
    save_to_excel(edges, rooms, xlsx_save)

    # ---------- 可视化检查 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(region_id_map, cmap='tab20')
    axes[0].set_title("Region ID Map")
    axes[1].imshow(wall_array, cmap='gray')
    axes[1].set_title("Wall Array")
    axes[2].imshow(icon_array, cmap='gray')
    axes[2].set_title("Icon Array")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
