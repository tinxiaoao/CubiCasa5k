import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
import networkx as nx


def build_topology_graph(rooms, edges):
    """
    修复后的根据房间列表和连接关系构建NetworkX图，允许同时记录门窗和墙连接。
    """
    G = nx.Graph()

    # 添加房间节点及属性
    for room in rooms:
        rid = room["id"]
        G.add_node(rid, room_type=room["room_type"], area=room["area"])

    # 添加边及属性，分别记录门窗和墙连接
    for (id1, id2), attr in edges.items():
        types = attr.get("connection_types", set())

        if {"door", "window", "door/window"} & types:
            G.add_edge(id1, id2, connection_type="door/window",
                       connection_count=attr["num_door_window"],
                       connection_area=attr["area_door_window"])

        if "wall" in types:
            G.add_edge(id1, id2, connection_type="wall",
                       connection_count=attr["num_wall"],
                       connection_area=attr["area_wall"])

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
    # 修复后的绘制连线部分
    # 最终修复后的拓扑图绘制逻辑，避免使用elif和默认黑色
    for (id1, id2), attr in edges.items():
        if id1 not in pos or id2 not in pos:
            continue
        x1, y1 = pos[id1]
        x2, y2 = pos[id2]
        types = set(attr.get("connection_types", []))

        # 同时存在门窗和墙连接时绘制莫兰迪粉蓝色
        if {"door", "window", "door/window"} & types and "wall" in types:
            color = (145, 168, 209, 255)  # 莫兰迪蓝
            draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
            continue

        # 仅存在门窗连接时绘制莫兰迪粉红色
        if {"door", "window", "door/window"} & types:
            color = (236, 179, 184, 255)  # 莫兰迪粉红
            draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
            continue

        # 仅存在墙连接时绘制莫兰迪灰色
        if "wall" in types:
            color = (173, 175, 170, 255)  # 莫兰迪灰
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
    room_type_map = {room['id']: room['room_type'] for room in rooms}
    data = []

    for (id1, id2), attr in edges.items():
        # 分别判断和记录门窗连接
        if attr.get("num_door_window", 0) > 0:
            data.append({
                "房间ID1": id1,
                "房间类型1": room_type_map.get(id1, "Unknown"),
                "房间ID2": id2,
                "房间类型2": room_type_map.get(id2, "Unknown"),
                "连接类型": "门/窗",
                "连接数量": attr["num_door_window"],
                "连接面积": attr["area_door_window"]
            })

        # 分别判断和记录墙连接
        if attr.get("num_wall", 0) > 0:
            data.append({
                "房间ID1": id1,
                "房间类型1": room_type_map.get(id1, "Unknown"),
                "房间ID2": id2,
                "房间类型2": room_type_map.get(id2, "Unknown"),
                "连接类型": "墙",
                "连接数量": attr["num_wall"],
                "连接面积": attr["area_wall"]
            })

    columns = ["房间ID1", "房间类型1", "房间ID2", "房间类型2", "连接类型", "连接数量", "连接面积"]
    df = pd.DataFrame(data, columns=columns)
    df.to_excel(save_path, index=False)
