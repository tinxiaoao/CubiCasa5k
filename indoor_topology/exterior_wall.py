import numpy as np
from typing import List, Dict, Tuple, Set

from indoor_topology.detect_adjacency import _get_boundary_pixels

# 常量
# ----------------------------
_DIRECTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4‑邻域
_WALL_CODE: int = 2  # palette 中: 墙体
_BLOCKED_CODES: Set[int] = {0, 1, 8, 50}  # palette 中: 室外 / 阻断


def _shoot_ray_to_outside(region_map: np.ndarray,
                          palette_map: np.ndarray,
                          start: Tuple[int, int],
                          direction: Tuple[int, int],
                          self_id: int,
                          max_wall_thickness: int):
    """
    仅用于外墙检测：
        • 若射线穿墙后直接撞外界，返回 wall_pixels 集合；
        • 否则返回 None。
    """
    H, W = region_map.shape
    dx, dy = direction
    x, y = start[0] + dx, start[1] + dy

    if not (0 <= x < H and 0 <= y < W):
        return None
    if region_map[x, y] == self_id:
        return None

    wall_pixels: Set[Tuple[int, int]] = set()
    wall_cnt = 0

    while 0 <= x < H and 0 <= y < W:
        rid = region_map[x, y]
        pal = palette_map[x, y]

        # 一旦遇到外界
        if rid == 0 and pal in _BLOCKED_CODES and pal != _WALL_CODE:
            return wall_pixels if wall_pixels else None  # 必须穿过墙

        # 墙体
        if pal == _WALL_CODE:
            wall_pixels.add((x, y))
            wall_cnt += 1
            if wall_cnt >= max_wall_thickness:
                return None  # 墙太厚

        # 遇到其他房间 / 阻断而非外界
        if rid != 0:
            return None
        if pal in _BLOCKED_CODES and pal != _WALL_CODE:
            return None

        x += dx
        y += dy
    return None


def exterior_wall_pixels(region_map: np.ndarray,
                         palette_map: np.ndarray,
                         max_wall_thickness: int) -> Dict[int, Tuple[int, int]]:
    """
    返回 {room_id: (length, width)} 外墙尺寸字典
    """
    result: Dict[int, Tuple[int, int]] = {}
    for rid in [i for i in np.unique(region_map) if i > 0]:
        ext_pix: Set[Tuple[int, int]] = set()
        for (x, y) in _get_boundary_pixels(region_map, rid):
            for dx, dy in _DIRECTIONS:  # 4 向
                pix = _shoot_ray_to_outside(region_map, palette_map,
                                            (x, y), (dx, dy),
                                            self_id=rid,
                                            max_wall_thickness=max_wall_thickness)
                if pix:
                    ext_pix.update(pix)
        if ext_pix:
            coords = np.array(list(ext_pix))
            h = coords[:, 0].ptp() + 1
            w = coords[:, 1].ptp() + 1
            result[rid] = (max(h, w), min(h, w))
    return result  # 每房间至多一条
