import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from indoor_topology.extract_rooms import extract_rooms
from indoor_topology.detect_adjacency_v2 import detect_adjacency
from indoor_topology.save_topology_image import save_topology_image, save_to_excel, build_topology_graph


def main():
    # 文件路径定义
    wall_svg_path = "wall_svg.png"
    icon_svg_path = "icon_svg.png"
    rough_image_path = "svgImg_roughcast.png"  # 补充原始roughcast图
    palette_img_path = "icon.png"  # 补充调色板图

    # 加载图像文件
    wall_label_img = Image.open(wall_svg_path).convert('P')
    wall_label_array = np.array(wall_label_img)

    # 提取房间信息
    region_id_map, rooms = extract_rooms(wall_label_array)

    # 提取墙体区域 (假设墙的索引是2)
    wall_array = (wall_label_array == 2).astype(np.uint8)

    # 加载icon图标文件 (门窗)
    icon_label_img = Image.open(icon_svg_path).convert('P')
    icon_label_array = np.array(icon_label_img)
    icon_array = np.zeros_like(icon_label_array, dtype=np.uint8)
    icon_array[icon_label_array == 1] = 1  # 门
    icon_array[icon_label_array == 2] = 2  # 窗

    # 检测邻接关系
    edges = detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array)

    # 构建拓扑图
    # topology_graph = build_topology_graph(rooms, edges)

    # 加载额外所需的图像参数
    rough_image = Image.open(rough_image_path).convert('RGBA')
    palette_img = Image.open(palette_img_path).convert('P')

    # 调用save_topology_image（补全所有必要参数）
    save_topology_image(
        region_id_map=region_id_map,
        wall_array=wall_array,
        rooms=rooms,
        edges=edges,
        save_path='debug_topology_result.png',
        rough_image=rough_image,
        palette_img=palette_img,
        wall_label_img=wall_label_img
    )
    print("拓扑图保存成功为：debug_topology_result.png")

    # 保存edges邻接关系为Excel
    save_to_excel(
        rooms=rooms,
        edges=edges,
        save_path='debug_topology_edges.xlsx'
    )
    print("拓扑邻接关系Excel表保存成功为：debug_topology_edges.xlsx")

    # 可视化检查
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(region_id_map, cmap='tab20')
    axes[0].set_title("Region ID Map (房间标记图)")
    axes[1].imshow(wall_array, cmap='gray')
    axes[1].set_title("Wall Array (墙体)")
    axes[2].imshow(icon_array, cmap='gray')
    axes[2].set_title("Icon Array (门窗)")
    plt.tight_layout()
    plt.show()

    # 打印edges信息确认
    print("\n详细邻接关系 (edges)：")
    for room_pair, info in edges.items():
        print(f"{room_pair}: {info}")


if __name__ == "__main__":
    main()
