import multiprocessing as mp
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from functools import partial

from indoor_topology.detect_adjacency import detect_adjacency
from indoor_topology.extract_rooms import extract_rooms
from indoor_topology.save_topology_image import save_topology_image, save_to_excel

# 数据根目录和文件
ORIGIN_ROOT = r"E:\code\floor_data\cubicasa5k"
TRAIN_LIST = os.path.join(ORIGIN_ROOT, "train.txt")

# 排除样本
EXCLUDE = [
    "high_quality_architectural/2003/", "high_quality_architectural/2565/",
    "high_quality_architectural/6143/", "high_quality_architectural/10074/",
    "high_quality_architectural/10754/", "high_quality_architectural/10769/",
    "high_quality_architectural/14611/", "high_quality/7092/",
    "high_quality/1692/", "high_quality_architectural/10"
]
EXCLUDE = [p.strip("/").lower() for p in EXCLUDE]

# 输出目录
OUT_ROOT = r"E:\code\CubiCasa5k\output"
TOP_DIR = os.path.join(OUT_ROOT, "topology")
XLSX_DIR = os.path.join(OUT_ROOT, "topology_excel")
os.makedirs(TOP_DIR, exist_ok=True)
os.makedirs(XLSX_DIR, exist_ok=True)

# -------- 调色板统一 --------
ICON_PAL = r"E:\code\CubiCasa5k\icon.png"
palette_img = Image.open(ICON_PAL).convert('P')
pal = palette_img.getpalette()
# 调整示例色板 (可选)
pal[50 * 3:50 * 3 + 3] = [255, 255, 255]
pal[35 * 3:35 * 3 + 3] = [160, 160, 160]
palette_img.putpalette(pal)


def process_sample(sample_dir: str):
    wall_path = os.path.join(sample_dir, "wall_svg.png")
    icon_path = os.path.join(sample_dir, "icon_svg.png")
    rough_path = os.path.join(sample_dir, "svgImg_roughcast.png")

    wall_label_img = Image.open(wall_path).convert('P')
    wall_label_img.putpalette(palette_img.getpalette())
    wall_label_array = np.array(wall_label_img)

    icon_label_img = Image.open(icon_path).convert('P')
    icon_label_img.putpalette(palette_img.getpalette())
    icon_label_array = np.array(icon_label_img)

    wall_array = (wall_label_array == 2).astype(np.uint8)
    icon_array = np.zeros_like(icon_label_array, dtype=np.uint8)
    icon_array[icon_label_array == 1] = 2
    icon_array[icon_label_array == 2] = 1

    region_id_map, rooms, min_len = extract_rooms(wall_label_array)
    edges = detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array, min_len)

    # 保存
    name = os.path.basename(os.path.normpath(sample_dir))
    save_topology_image(region_id_map, rooms, edges, rough_path,
                        os.path.join(TOP_DIR, f"{name}.png"))
    save_to_excel(edges, rooms, os.path.join(XLSX_DIR, f"{name}.xlsx"))
    return True


if __name__ == "__main__":
    # 读取样本路径
    with open(TRAIN_LIST, "r") as f:
        subs = [ln.strip().lstrip("\\/") for ln in f if ln.strip()]
    samples = [os.path.normpath(os.path.join(ORIGIN_ROOT, s)) for s in subs
               if not any(ex in s.lower() for ex in EXCLUDE)]

    worker = partial(process_sample)
    cpu_cnt = max(mp.cpu_count() - 2, 1)
    with mp.Pool(cpu_cnt) as pool:
        list(tqdm(pool.imap(worker, samples), total=len(samples)))
    print("处理完成样本数:", len(samples))
