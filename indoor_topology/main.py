import multiprocessing as mp
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from functools import partial

from indoor_topology.detect_adjacency import detect_adjacency
from indoor_topology.extract_rooms import extract_rooms
from indoor_topology.save_topology_image import save_topology_image, save_to_excel, build_topology_graph

# 数据根目录和文件
original_root = r"E:\\code\\floor_data\\cubicasa5k"
train_list_path = os.path.join(original_root, "train.txt")

# 排除样本子目录列表（使用统一格式便于匹配）
exclude_list = [
    "\\high_quality_architectural\\2003\\", "\\high_quality_architectural\\2565\\",
    "\\high_quality_architectural\\6143\\", "\\high_quality_architectural\\10074\\",
    "\\high_quality_architectural\\10754\\", "\\high_quality_architectural\\10769\\",
    "\\high_quality_architectural\\14611\\", "\\high_quality\\7092\\",
    "\\high_quality\\1692\\", "high_quality_architectural\\10"
]
exclude_list = [path.replace("\\", "/").lower().strip("/") for path in exclude_list]

# 读取 train.txt 并过滤路径
with open(train_list_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
sample_paths = []
for line in lines:
    subdir = line.lstrip("\\/")
    full_path = os.path.normpath(os.path.join(original_root, subdir))
    check_path = subdir.replace("\\", "/").lower()
    if any(excl in check_path for excl in exclude_list):
        continue
    sample_paths.append(full_path)

# 输出目录
output_root = r"E:\\code\\CubiCasa5k\\output"
output_topology_dir = os.path.join(output_root, "topology")
output_excel_dir = os.path.join(output_root, "topology_excel")
os.makedirs(output_topology_dir, exist_ok=True)
os.makedirs(output_excel_dir, exist_ok=True)


def process_sample(sample_dir, palette_img):

    wall_path = os.path.join(sample_dir, "wall_svg.png")
    icon_path = os.path.join(sample_dir, "icon_svg.png")
    rough_path = os.path.join(sample_dir, "svgImg_roughcast.png")

    wall_label_img = Image.open(wall_path).convert('P')
    wall_label_img.putpalette(palette_img.getpalette())
    wall_label_array = np.array(wall_label_img)

    icon_label_img = Image.open(icon_path).convert('P')
    icon_label_img.putpalette(palette_img.getpalette())
    icon_label_array = np.array(icon_label_img)

    # 直接根据原始索引定义墙体和门窗
    wall_array = (wall_label_array == 2).astype(np.uint8)
    icon_array = np.zeros_like(icon_label_array, dtype=np.uint8)
    icon_array[icon_label_array == 1] = 1  # 门
    icon_array[icon_label_array == 2] = 2  # 窗

    rough_image = Image.open(rough_path).convert("RGBA")

    region_id_map, rooms = extract_rooms(wall_label_array)
    edges = detect_adjacency(region_id_map, wall_array, icon_array)

    # ---------- 构图 ----------
    # 节点（Nodes） 代表房间，每个节点包含了房间的类型 (room_type) 和面积 (area) 等属性。
    # 边（Edges） 代表房间之间的连接关系，每条边记录了连接的类型（connection_type）、连接的数量（connection_count）和连接面积（connection_area）
    # G = build_topology_graph(rooms, edges)  # G暂且没用到，绘制拓扑

    # ---------- 保存 ----------
    base = os.path.basename(os.path.normpath(sample_dir))
    img_save_path = os.path.join(output_topology_dir, f"{base}.png")
    excel_save_path = os.path.join(output_excel_dir, f"{base}.xlsx")

    save_topology_image(
        region_id_map, wall_array, rooms, edges,
        img_save_path, rough_image,
        palette_img, wall_label_img)

    save_to_excel(rooms, edges, excel_save_path)
    return True


if __name__ == "__main__":
    # -------- 统一调色板 --------
    ICON_PALETTE_PATH = r"E:\code\CubiCasa5k\icon.png"
    palette_img = Image.open(ICON_PALETTE_PATH).convert('P')
    pal = palette_img.getpalette()
    pal[50 * 3:50 * 3 + 3] = [255, 255, 255]
    pal[35 * 3:35 * 3 + 3] = [160, 160, 160]
    palette_img.putpalette(pal)

    worker = partial(process_sample, palette_img=palette_img)

    cpu_cnt = max(mp.cpu_count() - 2, 1)
    with mp.Pool(cpu_cnt) as pool:
        for _ in tqdm(pool.imap(worker, sample_paths), total=len(sample_paths)):
            pass

    print("处理完成样本数:", len(sample_paths))
