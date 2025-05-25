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
            neighbor_ids = np.unique(neighbor_ids)  # sort and unique
            if neighbor_ids.size == 2:
                a, b = int(neighbor_ids[0]), int(neighbor_ids[1])
                key = ensure_edge_entry(a, b)
                edges[key]['connection_types'].add(icon_type)  # 'door' or 'window'
                edges[key]['num_door_window'] += 1
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **Wall connections**
    # 提取所有房间的ID（排除背景ID如0）
    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]

    # 创建记录墙连接的访问标记数组，避免重复计算
    visited_wall = np.zeros_like(wall_array, dtype=bool)

    # 排除类别像素ID集合（如0背景、1室外、2阳台等）
    excluded_categories = {0, 1, 8,}

    # 遍历每个房间，检查其边界像素的墙连接
    for rid in region_ids:
        # 获取房间rid的边界掩膜（边界像素为True）
        boundary_mask = get_boundary_mask(region_id_map, rid)
        boundary_coords = np.transpose(np.nonzero(boundary_mask))

        # 遍历该房间的每个边界像素
        for x, y in boundary_coords:
            # 检查四个正交方向的邻居像素
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # 跳过越界或非墙体的邻居
                if nx < 0 or nx >= region_id_map.shape[0] or ny < 0 or ny >= region_id_map.shape[1]:
                    continue
                if wall_array[nx, ny] != 1:
                    continue
                # 邻居是墙体，且尚未被访问过，开始沿该方向逐像素扫描
                if visited_wall[nx, ny]:
                    continue
                t = 1
                wall_pixels_segment = []  # 记录当前墙段的所有墙体像素
                encountered_room = None  # 保存扫描过程中遇到的另一房间ID
                last_wall_pixel = None  # 记录墙体穿透结束时的最后一个墙像素坐标
                while True:
                    cx = x + dx * t
                    cy = y + dy * t
                    # 超出边界则停止扫描
                    if cx < 0 or cx >= region_id_map.shape[0] or cy < 0 or cy >= region_id_map.shape[1]:
                        break
                    if wall_array[cx, cy] == 1:
                        # 仍在墙体内部，继续向前扩展
                        wall_pixels_segment.append((cx, cy))
                        t += 1
                        continue
                    else:
                        # 碰到非墙体像素，结束扫描
                        rid2 = region_id_map[cx, cy]
                        if rid2 != 0 and rid2 != rid:
                            # 找到另一个房间像素，记录连接的房间ID和最后一个墙体像素
                            encountered_room = rid2
                            last_wall_pixel = (cx - dx, cy - dy)
                        break
                # 若未发现另一房间，则不是房间间墙连接
                if encountered_room is None:
                    continue

                # 确定房间对 (rid, encountered_room)，按大小排序作为无向键
                id1, id2 = rid, encountered_room
                pair = (id1, id2) if id1 < id2 else (id2, id1)

                # 将扫描经过的墙体像素标记为已访问，避免重复识别
                for wx, wy in wall_pixels_segment:
                    visited_wall[wx, wy] = True

                # 根据扫描方向确定墙段走向（垂直或水平）及房间所在侧向，用于沿墙长度扩展
                if dx == 0 and dy == 1:  # 向东扫描，墙体走向为垂直（上下扩展）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, -1), (0, 1)  # A在墙的西侧，B在墙的东侧
                elif dx == 0 and dy == -1:  # 向西扫描，墙体走向为垂直（上下扩展）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, 1), (0, -1)  # A在东侧，B在西侧
                elif dx == 1 and dy == 0:  # 向南扫描，墙体走向为水平（左右扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (-1, 0), (1, 0)  # A在墙的北侧，B在墙的南侧
                elif dx == -1 and dy == 0:  # 向北扫描，墙体走向为水平（左右扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (1, 0), (-1, 0)  # A在南侧，B在北侧
                else:
                    orient_dirs = []

                # 选取穿透扫描结束时的最后墙体像素作为基准点
                base_x, base_y = last_wall_pixel if last_wall_pixel is not None else wall_pixels_segment[-1]

                # 分别向墙段两端（orient_dirs方向）扩展，获取整个连续墙段
                for odx, ody in orient_dirs:
                    curx, cury = base_x, base_y
                    while True:
                        curx += odx
                        cury += ody
                        # 超出边界或非墙体则停止延伸
                        if curx < 0 or curx >= region_id_map.shape[0] or cury < 0 or cury >= region_id_map.shape[1]:
                            break
                        if wall_array[curx, cury] != 1:
                            break
                        # 若墙体像素已记录过，表示该段墙已处理，停止延伸
                        if visited_wall[curx, cury]:
                            break
                        # 检查该墙体像素两侧是否仍然是对应的两个房间
                        ax, ay = curx + sideA[0], cury + sideA[1]  # A房间一侧邻接像素
                        bx, by = curx + sideB[0], cury + sideB[1]  # B房间另一侧邻接像素
                        if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                            break
                        if bx < 0 or bx >= region_id_map.shape[0] or by < 0 or by >= region_id_map.shape[1]:
                            break
                        # 如果当前墙像素两侧不再同时邻接房间A和房间B，表示墙段在此结束
                        if region_id_map[ax, ay] != rid or region_id_map[bx, by] != encountered_room:
                            break
                        # 墙段在该方向上继续，记录像素并标记访问
                        wall_pixels_segment.append((curx, cury))
                        visited_wall[curx, cury] = True

                # 统计该墙段的像素数和记录连接信息
                segment_pixels = set(wall_pixels_segment)  # 去重，获取该墙段所有独立像素
                segment_length = len(segment_pixels)

                # 检查该墙段是否邻接排除类别的像素，若是则视为外墙连接，跳过此墙段
                segment_mask = np.zeros_like(region_id_map, dtype=bool)
                for wx, wy in segment_pixels:
                    segment_mask[wx, wy] = True
                dilated_segment = binary_dilation(segment_mask, structure=structure, border_value=0)
                neighbor_area = dilated_segment & ~segment_mask
                neighbor_ids = np.unique(region_id_map[neighbor_area])
                # 若墙段任一像素邻接到排除类别（背景、室外、阳台等），跳过该墙段
                if np.any(np.isin(neighbor_ids, list(excluded_categories))):
                    continue

                # 初始化或更新edges字典中房间对的墙连接信息
                if pair not in edges:
                    edges[pair] = {
                        'connection_types': {'wall'},
                        'num_wall': 1,
                        'area_wall': segment_length,
                        'num_door_window': 0,
                        'area_door_window': 0
                    }
                else:
                    # 保留已有的连接类型，新增'wall'
                    edges[pair]['connection_types'].add('wall')
                    # 增加墙段数量和像素面积
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += segment_length

    return edges
