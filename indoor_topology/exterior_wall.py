import numpy as np
from typing import List, Dict, Tuple, Set

from indoor_topology.detect_adjacency import _get_boundary_pixels

_DIRECTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_WALL_CODE: int = 2
_BLOCKED_CODES: Set[int] = {0, 1, 8, 50}


def _shoot_ray_to_outside(region_map: np.ndarray,
                          palette_map: np.ndarray,
                          start: Tuple[int, int],
                          direction: Tuple[int, int],
                          self_id: int,
                          max_wall_thickness: int):
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

        if rid == 0 and pal in _BLOCKED_CODES and pal != _WALL_CODE:
            return wall_pixels if wall_pixels else None

        if pal == _WALL_CODE:
            wall_pixels.add((x, y))
            wall_cnt += 1
            if wall_cnt >= max_wall_thickness:
                return None

        if rid != 0:
            return None
        if pal in _BLOCKED_CODES and pal != _WALL_CODE:
            return None

        x += dx
        y += dy

    # 有些外墙紧贴着图像边缘，不加入这一步会漏掉检测这部分外墙
    if wall_pixels:
        return wall_pixels

    return None


def exterior_wall_segments(region_map: np.ndarray,
                           palette_map: np.ndarray,
                           max_wall_thickness: int) -> Dict[int, List[Tuple[str, int, int]]]:
    """
    返回 {room_id: [(direction, length, thickness), ...]} 外墙分段尺寸字典，合并同方向同厚度的墙段，
    并删除厚度大于长度的异常段。
    """
    result: Dict[int, List[Tuple[str, int, int]]] = {}
    for rid in [i for i in np.unique(region_map) if i > 0]:
        segments = []
        visited = set()

        for (x, y) in _get_boundary_pixels(region_map, rid):
            for direction in _DIRECTIONS:
                pix = _shoot_ray_to_outside(region_map, palette_map,
                                            (x, y), direction, rid, max_wall_thickness)
                if pix:
                    pix_frozen = frozenset(pix)
                    if pix_frozen in visited:
                        continue
                    visited.add(pix_frozen)
                    coords = np.array(list(pix))

                    dx, dy = direction
                    if dx != 0:
                        wall_dir = 'horizontal'
                        length = coords[:, 1].ptp() + 1
                        thickness = coords[:, 0].ptp() + 1
                    else:
                        wall_dir = 'vertical'
                        length = coords[:, 0].ptp() + 1
                        thickness = coords[:, 1].ptp() + 1

                    # 合并相同方向与厚度的外墙
                    merged = False
                    for i, (existing_dir, existing_length, existing_thickness) in enumerate(segments):
                        if existing_dir == wall_dir and existing_thickness == thickness:
                            segments[i] = (existing_dir, existing_length + length, existing_thickness)
                            merged = True
                            break
                    if not merged:
                        segments.append((wall_dir, length, thickness))

        # 删除厚度大于长度的异常段
        segments = [seg for seg in segments if seg[2] <= seg[1]]

        result[rid] = segments

    return result
