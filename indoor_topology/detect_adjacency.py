import numpy as np
import cv2
from typing import List, Dict, Tuple, Set

# ----------------------------
# detect_adjacency.py
# ----------------------------
# 说明：
#   本文件整合了门/窗检测 + "墙体隔墙" 检测 + "开敞相邻" 检测，
#   对外只暴露 detect_adjacency(...) 单一接口。
#   debug_topology.py 可直接调用：
#       edges = detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array)
#   返回值 edges 为 List[Dict]，字段说明与原始版本一致。
# ----------------------------

# 8 邻域方向 (dx, dy)
_DIRECTIONS: List[Tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1)
]
_WALL_CODE: int = 2  # 调色板索引中：墙体编码
_BLOCKED_CODES: Set[int] = {0, 1, 8, 50}  # 调色板索引中：室外/阻断编码


# ======================================================================
# 内部辅助函数（墙体隔墙检测专用）
# ======================================================================

def _get_boundary_pixels(region_map: np.ndarray, room_id: int) -> List[Tuple[int, int]]:
    """返回 room_id 在 region_map 中的所有边界像素坐标列表。"""
    H, W = region_map.shape
    coords = np.argwhere(region_map == room_id)
    boundary = []
    for x, y in coords:
        for dx, dy in _DIRECTIONS:
            nx, ny = x + dx, y + dy
            if nx < 0 or nx >= H or ny < 0 or ny >= W or region_map[nx, ny] != room_id:
                boundary.append((x, y))
                break
    return boundary


def _shoot_ray(
        region_map: np.ndarray,
        palette_map: np.ndarray,
        start: Tuple[int, int],
        direction: Tuple[int, int],
        self_id: int
):
    """沿 direction 发射射线。若穿墙到达另一房间返回 (target_id, wall_pixels_set)，否则 None。"""
    H, W = region_map.shape
    dx, dy = direction
    x, y = start[0] + dx, start[1] + dy

    # 起始一步检测
    if x < 0 or x >= H or y < 0 or y >= W:
        return None
    if region_map[x, y] == self_id:
        return None  # 切向或内向
    if region_map[x, y] == 0 and palette_map[x, y] in _BLOCKED_CODES and palette_map[x, y] != _WALL_CODE:
        return None  # 首步就是阻断

    wall_pixels: Set[Tuple[int, int]] = set()

    while True:
        if x < 0 or x >= H or y < 0 or y >= W:
            return None  # 出界

        rid = region_map[x, y]
        pal = palette_map[x, y]

        if rid != 0:  # 撞到房间
            if rid == self_id:
                return None  # 凹形回到自己
            if wall_pixels:
                return rid, wall_pixels
            return None  # 未穿墙直接到达 → 非墙连接

        # 非房间
        if pal in _BLOCKED_CODES:
            return None  # 阻断
        if pal == _WALL_CODE:
            wall_pixels.add((x, y))
        # 继续前进
        x += dx
        y += dy


def _wall_edges(region_map: np.ndarray, palette_map: np.ndarray) -> List[Dict]:
    """计算所有房间通过墙体连接的 edges 列表。"""
    room_ids = [rid for rid in np.unique(region_map) if rid > 0]
    edge_map: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}

    for rid in room_ids:
        boundary = _get_boundary_pixels(region_map, rid)
        for (x, y) in boundary:
            for dx, dy in _DIRECTIONS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < region_map.shape[0] and 0 <= ny < region_map.shape[1]:
                    if region_map[nx, ny] == rid:
                        continue  # 内向
                else:
                    continue  # 越界方向

                res = _shoot_ray(region_map, palette_map, (x, y), (dx, dy), rid)
                if res is None:
                    continue
                tgt_id, wall_pixels = res
                if not wall_pixels:
                    continue
                pair = (rid, tgt_id) if rid < tgt_id else (tgt_id, rid)
                edge_map.setdefault(pair, set()).update(wall_pixels)

    # 生成 edge 字典列表
    walls_edges: List[Dict] = []
    for (a, b), pixels in edge_map.items():
        if not pixels:
            continue
        px = np.array(list(pixels))
        min_r, min_c = px.min(axis=0)
        max_r, max_c = px.max(axis=0)
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        length = max(height, width)
        thick = min(height, width)
        walls_edges.append({
            'roomA': int(a),
            'roomB': int(b),
            'type': 'wall',
            'length': int(length),
            'width': int(thick)
        })
    return walls_edges


# ======================================================================
# Public API
# ======================================================================

def detect_adjacency(region_id_map: np.ndarray,
                     wall_array: np.ndarray,
                     icon_array: np.ndarray,
                     wall_label_array: np.ndarray):
    """室内拓扑提取主函数，返回包含 door / window / wall / opening 的 edges 列表。"""

    # ---- 复制以免修改原数据 ----
    region_id_map = np.array(region_id_map)
    wall_array = np.array(wall_array)
    icon_array = np.array(icon_array)
    wall_label_array = np.array(wall_label_array)  # palette 索引图

    edges: List[Dict] = []

    # ==================================================================
    # 1️⃣ 门窗连接检测 (与旧版保持一致)
    # ==================================================================
    door_mask = (icon_array == 1).astype(np.uint8)
    window_mask = (icon_array == 2).astype(np.uint8)

    def _process_icon_component(mask: np.ndarray, icon_type: str):
        num_labels, labels = cv2.connectedComponents(mask, connectivity=4)
        for lbl in range(1, num_labels):
            coords = np.where(labels == lbl)
            if coords[0].size == 0:
                continue
            rows, cols = coords
            neighbor_rooms: Set[int] = set()
            for r, c in zip(rows, cols):
                # 上下左右 4 邻域
                if r > 0 and region_id_map[r - 1, c] > 0: neighbor_rooms.add(int(region_id_map[r - 1, c]))
                if r < region_id_map.shape[0] - 1 and region_id_map[r + 1, c] > 0: neighbor_rooms.add(
                    int(region_id_map[r + 1, c]))
                if c > 0 and region_id_map[r, c - 1] > 0: neighbor_rooms.add(int(region_id_map[r, c - 1]))
                if c < region_id_map.shape[1] - 1 and region_id_map[r, c + 1] > 0: neighbor_rooms.add(
                    int(region_id_map[r, c + 1]))
            if len(neighbor_rooms) == 2:
                roomA, roomB = sorted(list(neighbor_rooms))
                min_r, max_r = rows.min(), rows.max()
                min_c, max_c = cols.min(), cols.max()
                height = max_r - min_r + 1
                width = max_c - min_c + 1
                length = max(height, width)
                thick = min(height, width)
                edges.append({
                    'roomA': roomA,
                    'roomB': roomB,
                    'type': icon_type,
                    'length': int(length),
                    'width': int(thick)
                })

    _process_icon_component(door_mask, 'door')
    _process_icon_component(window_mask, 'window')

    # ==================================================================
    # 2️⃣ 墙体隔墙检测 (整合自 wall_edge_extractor.py)
    # ==================================================================
    wall_edges = _wall_edges(region_id_map, wall_label_array)
    edges.extend(wall_edges)

    # ==================================================================
    # 3️⃣ 开敞相邻检测
    # ==================================================================
    open_pair_points: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    H, W = region_id_map.shape
    for r in range(H):
        for c in range(W):
            rid = region_id_map[r, c]
            if rid <= 0:
                continue
            # 右邻
            if c < W - 1 and region_id_map[r, c + 1] > 0 and region_id_map[r, c + 1] != rid and wall_array[
                r, c + 1] == 0:
                pair = tuple(sorted((int(rid), int(region_id_map[r, c + 1]))))
                mark = (r, c) if pair[0] == rid else (r, c + 1)
                open_pair_points.setdefault(pair, []).append(mark)
            # 下邻
            if r < H - 1 and region_id_map[r + 1, c] > 0 and region_id_map[r + 1, c] != rid and wall_array[
                r + 1, c] == 0:
                pair = tuple(sorted((int(rid), int(region_id_map[r + 1, c]))))
                mark = (r, c) if pair[0] == rid else (r + 1, c)
                open_pair_points.setdefault(pair, []).append(mark)

    for pair, pts in open_pair_points.items():
        if not pts:
            continue
        mask = np.zeros_like(region_id_map, dtype=np.uint8)
        for pr, pc in pts:
            mask[pr, pc] = 1
        num_lbl, comp = cv2.connectedComponents(mask, connectivity=4)
        for lbl in range(1, num_lbl):
            coords = np.where(comp == lbl)
            if coords[0].size == 0:
                continue
            rows, cols = coords
            min_r, max_r = rows.min(), rows.max()
            min_c, max_c = cols.min(), cols.max()
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            length = max(height, width)
            thick = min(height, width)
            roomA, roomB = pair
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': 'opening',
                'length': int(length),
                'width': int(thick)
            })

    return edges
