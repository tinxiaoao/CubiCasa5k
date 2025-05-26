import numpy as np
from scipy.ndimage import binary_erosion, label, binary_dilation, binary_propagation


def get_boundary_mask(region_id_map, rid):
    """
    Returns a boolean mask of boundary pixels for room `rid`.
    A boundary pixel is one that belongs to room `rid` but has at least
    one direct neighbor (up, down, left, or right) that is not in `rid`.
    """
    # Create binary mask for the specified room id
    room_mask = (region_id_map == rid)
    if not room_mask.any():
        # If room id not present, return an all-false mask
        return np.zeros_like(region_id_map, dtype=bool)
    # Define a 4-connected structuring element (including center pixel)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    # Erode the room mask to get interior pixels (border_value=0 treats outside as empty)
    interior = binary_erosion(room_mask, structure=structure, border_value=0)
    # Boundary pixels are those in room_mask that were eroded away (not in interior)
    boundary_mask = room_mask & ~interior
    return boundary_mask


def detect_adjacency(region_id_map, wall_array, icon_array):
    """
    Detects adjacency between rooms via doors, windows, and walls.
    Returns a dict where keys are (room1, room2) tuples and values contain:
      - 'connection_types': set of {'door', 'window', 'wall'} indicating types of connections.
      - 'num_door_window': count of door/window openings between the rooms.
      - 'area_door_window': total area (in pixels) of those door/window openings.
      - 'num_wall': count of wall segments between the rooms.
      - 'area_wall': total area (pixels) of those wall segments.
    """
    edges = {}

    def ensure_edge_entry(a, b):
        """Helper to initialize dictionary entry for a room pair if not exists."""
        key = (a, b) if a < b else (b, a)
        if key not in edges:
            edges[key] = {
                'connection_types': set(),
                'num_door_window': 0,
                'area_door_window': 0,
                'num_wall': 0,
                'area_wall': 0
            }
        return key

    # 4-connected structure for labeling and dilation (cross-shaped)
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    # **Door and Window connections**
    for icon_val, icon_type in [(1, 'door'), (2, 'window')]:
        icon_mask = (icon_array == icon_val)
        labeled_icons, num_icons = label(icon_mask, structure=structure)
        for lbl in range(1, num_icons + 1):
            comp_mask = (labeled_icons == lbl)
            if not comp_mask.any():
                continue
            # Find unique neighboring room IDs by dilating the icon region
            dilated = binary_dilation(comp_mask, structure=structure, border_value=0)
            neighbor_area = dilated & ~comp_mask  # pixels adjacent to the icon region
            neighbor_ids = np.unique(region_id_map[neighbor_area])
            neighbor_ids = neighbor_ids[neighbor_ids > 0]  # exclude non-room (0)
            if neighbor_ids.size == 2:
                a, b = int(neighbor_ids[0]), int(neighbor_ids[1])
                key = ensure_edge_entry(a, b)
                edges[key]['connection_types'].add(icon_type)  # add 'door' or 'window'
                edges[key]['num_door_window'] += 1
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **Wall connections**
    # 使用连通性分析识别所有独立墙段
    wall_mask = (wall_array == 1)
    labeled_walls, num_segments = label(wall_mask, structure=structure)
    # 确定需要排除的类别索引并跳过与这些类别相邻的墙段 (判定为外墙)
    exclude_indices = {0, 1, 8, 50}
    # 遍历每个墙段，判断其相邻的房间ID集合
    for seg_label in range(1, num_segments + 1):
        seg_mask = (labeled_walls == seg_label)
        if not seg_mask.any():
            continue
        # 获取墙段周围相邻的所有区域ID
        dilated = binary_dilation(seg_mask, structure=structure, border_value=0)
        neighbor_area = dilated & ~seg_mask
        neighbor_ids = np.unique(region_id_map[neighbor_area])
        # 若墙段与需排除的类别相邻，则跳过 (外墙不计入房间间连接)
        if any(n in exclude_indices for n in neighbor_ids):
            continue
        # 排除背景ID 0，只保留房间ID
        neighbor_ids = neighbor_ids[neighbor_ids != 0]
        # 若少于两个房间相邻，则非房间间墙连接，跳过
        if neighbor_ids.size < 2:
            continue
        # 若墙段仅邻接两个房间，则计入一次墙连接
        if neighbor_ids.size == 2:
            id1, id2 = int(neighbor_ids[0]), int(neighbor_ids[1])
            key = ensure_edge_entry(id1, id2)
            edges[key]['connection_types'].add('wall')
            edges[key]['num_wall'] += 1
            edges[key]['area_wall'] += int(seg_mask.sum())
        else:
            # 若墙段邻接超过两个房间，则仅保留像素数最小的房间对连接
            pair_contact = {}
            wall_pixels = np.transpose(np.nonzero(seg_mask))
            for (x, y) in wall_pixels:
                # 检查当前墙像素左右相邻的房间
                if y - 1 >= 0 and y + 1 < region_id_map.shape[1]:
                    left_id = region_id_map[x, y - 1]
                    right_id = region_id_map[x, y + 1]
                    if left_id > 0 and right_id > 0 and left_id != right_id:
                        pid1, pid2 = int(left_id), int(right_id)
                        if pid1 > pid2:
                            pid1, pid2 = pid2, pid1
                        # 两侧均为有效房间ID才计入
                        if pid1 not in exclude_indices and pid2 not in exclude_indices:
                            pair_contact[(pid1, pid2)] = pair_contact.get((pid1, pid2), 0) + 1
                # 检查当前墙像素上下相邻的房间
                if x - 1 >= 0 and x + 1 < region_id_map.shape[0]:
                    up_id = region_id_map[x - 1, y]
                    down_id = region_id_map[x + 1, y]
                    if up_id > 0 and down_id > 0 and up_id != down_id:
                        pid1, pid2 = int(up_id), int(down_id)
                        if pid1 > pid2:
                            pid1, pid2 = pid2, pid1
                        if pid1 not in exclude_indices and pid2 not in exclude_indices:
                            pair_contact[(pid1, pid2)] = pair_contact.get((pid1, pid2), 0) + 1
            if not pair_contact:
                continue
            # 找出墙段连接面积（像素数）最小的房间对
            min_pair = None
            min_area = None
            for pr, count in pair_contact.items():
                if min_area is None or count < min_area:
                    min_area = count
                    min_pair = pr
            if min_pair is None:
                continue
            # 仅保留该房间对的墙连接（像素面积为最小面积）
            id1, id2 = min_pair
            key = ensure_edge_entry(id1, id2)
            edges[key]['connection_types'].add('wall')
            edges[key]['num_wall'] += 1
            edges[key]['area_wall'] += int(min_area)
    # 确保每段墙体连接最多只统计一次（通过上述逻辑已实现）
    return edges
