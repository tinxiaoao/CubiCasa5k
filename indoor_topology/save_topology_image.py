import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from typing import List, Dict, Tuple
import cv2

__all__ = [
    "build_topology_graph",
    "save_topology_image",
    "save_to_excel",
]


# --------------------------------------------------------------------------------------
# 1️⃣ 构图：将 edges 与 rooms 转为 NetworkX 图（仅供需要时使用，可不导入 networkx）
# --------------------------------------------------------------------------------------

def build_topology_graph(rooms: List[Dict], edges: List[Dict]):
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("build_topology_graph 依赖 networkx，请先 pip install networkx")

    G = nx.Graph()
    for r in rooms:
        G.add_node(r["id"], area=r["area"], room_type=r.get("room_type", ""))

    type_map = {"door": "门", "window": "窗", "wall": "墙", "opening": "开敞"}
    pair2types = defaultdict(set)
    for e in edges:
        pair = tuple(sorted((e["roomA"], e["roomB"])))
        pair2types[pair].add(type_map.get(e["type"], e["type"]))

    for (a, b), tset in pair2types.items():
        G.add_edge(a, b, types="|".join(sorted(tset)))
    return G


# --------------------------------------------------------------------------------------
# 2️⃣ 保存拓扑图 (PNG) —— 使用 roughcast PNG 作为背景；节点放置在房间质心
# --------------------------------------------------------------------------------------

def _compute_room_centroids(region_id_map: np.ndarray) -> Dict[int, Tuple[float, float]]:
    """返回房间ID → (y,x) 质心坐标 (以原图像坐标)"""
    centroids = {}
    ids = np.unique(region_id_map)
    ids = ids[ids > 0]
    for rid in ids:
        ys, xs = np.where(region_id_map == rid)
        if ys.size == 0:
            continue
        cy = ys.mean()
        cx = xs.mean()
        centroids[int(rid)] = (float(cy), float(cx))
    return centroids


def save_topology_image(region_id_map: np.ndarray,
                        rooms: List[Dict],
                        edges: List[Dict],
                        rough_img_path: str,
                        save_path: str,
                        font_path: str = None):
    """
    在 roughcast 背景图上绘制拓扑：
        • 每个节点绘制在房间质心位置
        • 边颜色: 门红 / 窗蓝 / 开敞绿 / 仅墙灰
        • 节点大小与房间面积 √ 成正比
    参数:
        region_id_map : (H,W) 房间ID矩阵
        rooms         : List[Dict] (需含 id, area)
        edges         : List[Dict]
        rough_img_path: 背景 PNG (svgImg_roughcast.png 渲染版)
        save_path     : 输出 PNG
    """
    # --- 1. 保护性检查 ---------------------------------
    if len(rooms) == 0:
        print(f"[WARN] {rough_img_path}: 无房间，跳过保存拓扑图")
        return  # 或者记录日志后 return

    # 读取背景
    bg = cv2.imread(rough_img_path, cv2.IMREAD_UNCHANGED)
    if bg is None:
        raise FileNotFoundError(rough_img_path)
    # 转 RGB
    if bg.shape[2] == 4:
        bg_rgba = cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA)
    else:
        bg_rgba = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    H, W = bg_rgba.shape[:2]

    # 计算质心
    centroids = _compute_room_centroids(region_id_map)

    # 正常化面积到半径
    areas = np.array([r["area"] for r in rooms])
    r_min, r_max = 6, 25
    radii = r_min + (np.sqrt(areas) - np.sqrt(areas).min()) / (np.sqrt(areas).ptp() + 1e-6) * (r_max - r_min)
    id2radius = {r["id"]: float(rad) for r, rad in zip(rooms, radii)}

    # 创建 RGBA 画布并叠加背景
    canvas = Image.fromarray(bg_rgba)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 先画边 (在节点下层)
    """颜色规则:
        1. 仅墙  -> 灰  (#888888)
        2. 同时包含墙 + (门/窗) -> 蓝 (#1f77b4)
        3. 其他(只有门/窗/开敞) -> 绿 (#2ca02c)  (保持原有门/窗/开敞混合情况)
    """
    pair2types = defaultdict(set)
    for e in edges:
        pair = tuple(sorted((e["roomA"], e["roomB"])))
        pair2types[pair].add(e["type"])

    for (a, b), tset in pair2types.items():
        if a not in centroids or b not in centroids:
            continue
        y1, x1 = centroids[a]
        y2, x2 = centroids[b]

        # 颜色判断
        has_wall = "wall" in tset
        has_doorwin = any(t in tset for t in ("door", "window"))
        if has_wall and has_doorwin:
            color = (31, 119, 180, 255)  # 蓝: 墙 + 门/窗
        elif has_wall:
            color = (136, 136, 136, 255)  # 仅墙: 灰
        else:
            color = (44, 160, 44, 255)  # 其他: 绿 (开敞或纯门窗)
        draw.line((x1, y1, x2, y2), fill=color, width=3)

    # 绘制节点
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    for r in rooms:
        rid = r["id"]
        if rid not in centroids:
            continue
        cy, cx = centroids[rid]
        rad = id2radius[rid]
        cx, cy, rad = float(cx), float(cy), float(rad)
        draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=(173, 216, 230, 200), outline=(0, 0, 0, 255),
                     width=2)
        txt = str(rid)
        tw, th = draw.textsize(txt, font=font)
        draw.text((cx - tw / 2, cy - th / 2), txt, fill=(0, 0, 0, 255), font=font)

    canvas.save(save_path)


# --------------------------------------------------------------------------------------
# 3️⃣  保存拓扑 Excel (单 sheet)
# --------------------------------------------------------------------------------------

def save_to_excel(edges: List[Dict], save_path: str):
    """列: 房间1 | 房间2 | 连接类型 | 数量 | length | width (支持墙体分段信息，合并同一房间对墙体记录)"""
    en = {"door": "Door", "window": "Window", "wall": "Wall", "opening": "Opening"}

    aggregated = {}
    for e in edges:
        key = (e["roomA"], e["roomB"], e["type"])
        if e["type"] == "wall" and "segments" in e:
            aggregated.setdefault(key, []).extend(e["segments"])
        else:
            aggregated.setdefault(key, []).append({"length": e["length"], "width": e["width"]})

    rows = []
    for (roomA, roomB, conn_type), segments in aggregated.items():
        row = {
            "RoomA": roomA,
            "RoomB": roomB,
            "ConnectionType": en.get(conn_type, conn_type),
            "Count": len(segments)
        }
        for i, seg in enumerate(segments, 1):
            row[f"Length_{i}"] = seg["length"]
            row[f"Width_{i}"] = seg["width"]
        rows.append(row)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(rows).to_excel(save_path, index=False)



# 4️⃣ 房间属性表

def save_rooms_excel(rooms: List[Dict], ext_stats: Dict[int, List[Tuple[str, int, int]]], out_path: str):
    """列: 房间ID | 类型 | 面积 | 周长 | 外墙方向 | 外墙长度 | 外墙厚度 | (多个外墙分段依次类推)"""
    rows = []
    for r in rooms:
        rid = r["id"]
        segments = ext_stats.get(rid, [])
        row_data = {
            "RoomID": rid,
            "Type": r.get("room_type", ""),
            "Area": r["area"],
            "Perimeter": r["perimeter"]
        }
        # 每段外墙单独增加列
        for idx, (wall_dir, length, thickness) in enumerate(segments, start=1):
            row_data[f"WallDir_{idx}"] = wall_dir
            row_data[f"ExtWallLength_{idx}"] = length
            row_data[f"ExtWallWidth_{idx}"] = thickness

        rows.append(row_data)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_excel(out_path, index=False)
