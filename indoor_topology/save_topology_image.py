import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import networkx as nx


def build_topology_graph(rooms, edges):

        G = nx.Graph()

        for room in rooms:
            G.add_node(room['id'], **room)

        for edge in edges:
            id1, id2, ctype = edge['roomA'], edge['roomB'], edge['type']
            if ctype == 'opening':
                connection_label = "开敞空间"
            elif ctype == 'door':
                connection_label = "门"
            elif ctype == 'window':
                connection_label = "窗"
            elif ctype == 'wall':
                connection_label = "墙"
            else:
                connection_label = "其他"

            G.add_edge(id1, id2, connection_type=connection_label,
                       length=edge.get('length', 0),
                       width=edge.get('width', 0))

        return G


def save_topology_image(region_id_map, wall_array, rooms, edges,
                        save_path, rough_image, palette_img, wall_label_img,
                        r_min=6, r_max=30, font_path=None):
    """
    在 rough_image 上绘制房间拓扑：
        1) 节点大小按房间面积自适应，限制在 [r_min, r_max] 像素；
        2) 节点颜色取自 wall_label_img 的 palette 索引；
        3) 边颜色：门/窗绿，墙灰，其余蓝。
    依赖全局变量：palette_img, wall_label_img
    """

    palette = palette_img.getpalette()
    img = rough_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    # ---------- 1. 计算质心 & 面积 ----------
    pos, areas, sqrt_areas = {}, {}, []
    for room in rooms:
        rid = room["id"]
        ys, xs = np.where(region_id_map == rid)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        pos[rid] = (cx, cy)
        areas[rid] = len(xs)
        sqrt_areas.append(math.sqrt(len(xs)))

    if not sqrt_areas:
        print("未找到房间，跳过绘制")
        return

    sq_min, sq_max = min(sqrt_areas), max(sqrt_areas)
    den = sq_max - sq_min if sq_max != sq_min else 1.0

    # ---------- 2. 绘制连线 ----------
    # 修正后的 edges 为列表结构
    for edge in edges:
        id1 = edge['roomA']
        id2 = edge['roomB']
        connection_type = edge['type']

        if id1 not in pos or id2 not in pos:
            continue

        x1, y1 = pos[id1]
        x2, y2 = pos[id2]

        # 提取同一房间对之间的所有连接类型
        room_pair = tuple(sorted((id1, id2)))
        types_set = set(e['type'] for e in edges if tuple(sorted((e['roomA'], e['roomB']))) == room_pair)

        # 只有 'door'/'window' 或者只有 'opening' 一种类型
        if types_set.issubset({'door', 'window'}) or types_set == {'opening'}:
            color = (236, 179, 184, 255)  # 莫兰迪粉红色

        # 只有 'wall' 类型
        elif types_set == {'wall'}:
            color = (173, 175, 170, 255)  # 莫兰迪灰色

        # 混合类型：wall + (door/window/opening)
        elif 'wall' in types_set and (types_set & {'door', 'window', 'opening'}):
            color = (145, 168, 209, 255)  # 莫兰迪蓝色

        draw.line([(x1, y1), (x2, y2)], fill=color, width=3)

    # ---------- 3. 绘制节点 ----------
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for rid, (cx, cy) in pos.items():
        # 半径映射
        r = int(r_min + (math.sqrt(areas[rid]) - sq_min) / den * (r_max - r_min))
        r = max(r_min, min(r, r_max * 2))

        # palette 取色
        idx = wall_label_img.getpixel((cx, cy))
        rgb = tuple(palette[idx * 3: idx * 3 + 3])
        fill_rgba = (*rgb, 255)

        # 画圆 + 黑描边
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(bbox, fill=fill_rgba, outline=(0, 0, 0, 255), width=2)

        # 文字颜色：亮底用黑，暗底用白
        bright = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt_col = (0, 0, 0) if bright > 128 else (255, 255, 255)
        draw.text((cx, cy), str(rid), fill=txt_col, font=font, anchor="mm")

    # ---------- 4. 保存 ----------
    img.save(save_path)


def save_to_excel(rooms, edges, save_path: str):
    """
    修复后的将房间属性和连接边列表保存到Excel文件的函数。
    允许同时记录门窗连接和墙连接。
    """
    data = []
    room_type_map = {room['id']: room['room_type'] for room in rooms}

    for edge in edges:
        id1 = edge['roomA']
        id2 = edge['roomB']
        connection_type = edge['type']

        if connection_type == 'door':
            ctype_cn = "门"
        elif connection_type == 'window':
            ctype_cn = "窗"
        elif connection_type == 'wall':
            ctype_cn = "墙"
        elif connection_type == 'opening':
            ctype_cn = "开敞空间"  # 新增的描述
        else:
            ctype_cn = "其他"

        data.append({
            "房间ID1": id1,
            "房间类型1": room_type_map.get(id1, "Unknown"),
            "房间ID2": id2,
            "房间类型2": room_type_map.get(id2, "Unknown"),
            "连接类型": ctype_cn,
            "连接长度": edge.get("length", 0),
            "连接宽度": edge.get("width", 0)
        })

    columns = ["房间ID1", "房间类型1", "房间ID2", "房间类型2", "连接类型", "连接长度", "连接宽度"]
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(save_path, index=False)

