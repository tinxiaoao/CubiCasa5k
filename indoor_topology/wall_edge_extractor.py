"""
wall_edge_extractor.py

提取房间之间的墙体连接并输出 Excel
"""

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set

# ---------- 参数 ----------
WALL_CODE = 2
BLOCKED_CODES = {0, 1, 8, 50}          # palette 中代表“室外/阻断”的编码（不含墙）
DIRECTIONS = [                         # 8 邻域方向 (dx, dy)
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1)
]

# ---------- 工具函数 ----------
def load_indexed_png(path: str) -> np.ndarray:
    """
    以“调色板索引”方式加载 png，返回 H×W 的二维数组（uint8/uint16）
    OpenCV 读入带调色板索引 png 会给出单通道数组，无需额外转换。
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim != 2:
        raise ValueError(f"{path} 不是单通道索引 PNG")
    return img

def get_boundary_pixels(region_map: np.ndarray, room_id: int) -> List[Tuple[int, int]]:
    """
    找出 room_id 在 region_map 中的所有边界像素坐标。
    """
    H, W = region_map.shape
    idx = np.argwhere(region_map == room_id)          # (N,2)
    boundary = []
    for x, y in idx:
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= H or ny < 0 or ny >= W:
                boundary.append((x, y))
                break
            if region_map[nx, ny] != room_id:
                boundary.append((x, y))
                break
    return boundary

def shoot_ray(region_map: np.ndarray,
              wall_map: np.ndarray,
              start: Tuple[int, int],
              direction: Tuple[int, int],
              self_id: int) -> Tuple[int, Set[Tuple[int, int]]] or None:
    """
    从 start 像素沿 direction 发射射线，返回
      (target_room_id, {墙体像素坐标集合})
    或 None（无墙连接）。
    """
    H, W = region_map.shape
    dx, dy = direction
    x, y = start[0] + dx, start[1] + dy

    # 若第一步就重回自身房间 → 切向；或碰到阻断 → 不延伸
    if x < 0 or x >= H or y < 0 or y >= W:
        return None
    if region_map[x, y] == self_id:
        return None
    if region_map[x, y] == 0 and wall_map[x, y] in BLOCKED_CODES and wall_map[x, y] != WALL_CODE:
        return None

    wall_pixels = set()

    while True:
        # 出界
        if x < 0 or x >= H or y < 0 or y >= W:
            return None

        rid = region_map[x, y]
        pal = wall_map[x, y]

        if rid != 0:            # 撞到房间
            if rid == self_id:
                return None     # 自己 -> 凹形回环，忽略
            if wall_pixels:      # 经过了墙才到对方
                return rid, wall_pixels
            # 若未跨墙就到另一房间，则是“直接邻接”(非墙隔)，不记
            return None

        # 非房间区域
        if pal in BLOCKED_CODES:
            return None
        if pal == WALL_CODE:
            wall_pixels.add((x, y))
        # 继续前进
        x += dx
        y += dy

def collect_wall_edges(region_map: np.ndarray,
                       wall_map: np.ndarray) -> pd.DataFrame:
    """
    扫描所有房间，汇总“墙体连接”的房间对 DataFrame
    """
    result: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    room_ids = [i for i in np.unique(region_map) if i != 0]

    for rid in room_ids:
        boundary = get_boundary_pixels(region_map, rid)
        for (x, y) in boundary:
            # 仅朝“离开房间”的方向射线
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= region_map.shape[0] or ny < 0 or ny >= region_map.shape[1]:
                    continue
                if region_map[nx, ny] == rid:
                    continue        # 该方向仍在房间内部
                res = shoot_ray(region_map, wall_map, (x, y), (dx, dy), rid)
                if res is None:
                    continue
                target_id, wall_pixels = res
                pair = (rid, target_id) if rid < target_id else (target_id, rid)
                result.setdefault(pair, set()).update(wall_pixels)

    records = [
        dict(房间ID1=a, 房间ID2=b, 连接类型="wall", 连接面积=len(pixels))
        for (a, b), pixels in result.items()
        if len(pixels) > 0
    ]
    return pd.DataFrame(records, columns=["房间ID1", "房间ID2", "连接类型", "连接面积"])

# ---------- 主入口 ----------
if __name__ == "__main__":
    # === 修改为实际文件路径 ===
    region_map_path = "region_id_map.png"   # 由 extract_rooms 保存的索引图
    wall_map_path   = "wall_svg.png"

    region_id_map = load_indexed_png(region_map_path)
    wall_label_img = load_indexed_png(wall_map_path)

    df = collect_wall_edges(region_id_map, wall_label_img)
    df.to_excel("wall_edges.xlsx", index=False)
    print(f"已输出 {len(df)} 条墙体连接到 wall_edges.xlsx")
