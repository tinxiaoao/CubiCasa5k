import numpy as np
from scipy.ndimage import binary_erosion, label, binary_dilation, binary_propagation
from PIL import Image


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


def detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array):
    """
    Detects adjacency between rooms via doors, windows, and walls.
    Now excludes "exterior walls" (walls adjacent to regions with labels in {0,1,8,50})
    from room-to-room wall connectivity.
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

    # **Door and Window connections** (保持原有逻辑)
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
                edges[key]['connection_types'].add(icon_type)  # 'door' or 'window'
                edges[key]['num_door_window'] += 1
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **Wall connections**
    # 使用 wall_label_array 判定外墙：排除 exclude_indices 区域邻接的墙体
    exclude_indices = {0, 1, 8, 50}
    exclude_mask = np.isin(wall_label_array, list(exclude_indices))
    # 将排除区域膨胀一圈，得到与这些区域相邻的所有像素
    exclude_neighbors = binary_dilation(exclude_mask, structure=structure, border_value=0)
    # 外墙掩码：与排除区域邻接且自身为墙体的像素
    external_wall_mask = exclude_neighbors & (wall_array == 1)
    # 内部墙体掩码（排除外墙后的墙体）
    internal_wall = wall_array.copy().astype(np.uint8)
    internal_wall[external_wall_mask] = 0
    # 调试输出：保存排除外墙后的 wall_array 掩码图像
    internal_wall_img = (internal_wall * 255).astype(np.uint8)
    Image.fromarray(internal_wall_img).save("debug_internal_wall_mask.png")

    # 提取所有房间的 ID（排除背景 ID 如 0）
    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]

    # 创建记录墙连接的访问标记数组，避免重复计算
    visited_wall = np.zeros_like(internal_wall, dtype=bool)

    # 遍历每个房间，检查其边界像素的墙连接
    for rid in region_ids:
        # 获取房间 rid 的边界掩膜（边界像素为 True）
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
                if internal_wall[nx, ny] != 1:
                    continue
                # 邻居是墙体，且尚未被访问过，开始沿该方向逐像素扫描
                if visited_wall[nx, ny]:
                    continue
                t = 1
                wall_pixels_segment = []  # 记录当前墙段的所有墙体像素
                encountered_room = None  # 保存扫描过程中遇到的另一房间 ID
                last_wall_pixel = None  # 记录墙体穿透结束时的最后一个墙像素坐标
                while True:
                    cx = x + dx * t
                    cy = y + dy * t
                    # 超出边界则停止扫描
                    if cx < 0 or cx >= region_id_map.shape[0] or cy < 0 or cy >= region_id_map.shape[1]:
                        break
                    if internal_wall[cx, cy] == 1:
                        # 仍在墙体内部，继续向前扩展
                        wall_pixels_segment.append((cx, cy))
                        t += 1
                        continue
                    else:
                        # 碰到非墙体像素，结束扫描
                        rid2 = region_id_map[cx, cy]
                        if rid2 != 0 and rid2 != rid:
                            # 找到另一个房间像素，记录连接的房间 ID 和最后一个墙体像素
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
                    sideA, sideB = (0, -1), (0, 1)  # A 在墙的西侧，B 在墙的东侧
                elif dx == 0 and dy == -1:  # 向西扫描，墙体走向为垂直（上下扩展）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, 1), (0, -1)  # A 在墙的东侧，B 在墙的西侧
                elif dx == 1 and dy == 0:  # 向南扫描，墙体走向为水平（左右扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (-1, 0), (1, 0)  # A 在墙的北侧，B 在墙的南侧
                elif dx == -1 and dy == 0:  # 向北扫描，墙体走向为水平（左右扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (1, 0), (-1, 0)  # A 在墙的南侧，B 在墙的北侧
                else:
                    orient_dirs = []

                # 选取穿透扫描结束时的最后墙体像素作为基准点
                base_x, base_y = last_wall_pixel if last_wall_pixel is not None else wall_pixels_segment[-1]

                # 分别向墙段两端（orient_dirs 方向）扩展，获取整个连续墙段
                for odx, ody in orient_dirs:
                    curx, cury = base_x, base_y
                    while True:
                        curx += odx
                        cury += ody
                        # 超出边界或非墙体则停止延伸
                        if curx < 0 or curx >= region_id_map.shape[0] or cury < 0 or cury >= region_id_map.shape[1]:
                            break
                        if internal_wall[curx, cury] != 1:
                            break
                        # 若墙体像素已记录过，表示该段墙已处理，停止延伸
                        if visited_wall[curx, cury]:
                            break
                        # 检查该墙体像素两侧是否仍然是对应的两个房间
                        ax, ay = curx + sideA[0], cury + sideA[1]  # A 房间一侧邻接像素
                        bx, by = curx + sideB[0], cury + sideB[1]  # B 房间另一侧邻接像素
                        if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                            break
                        if bx < 0 or bx >= region_id_map.shape[0] or by < 0 or by >= region_id_map.shape[1]:
                            break
                        # 如果当前墙像素两侧不再同时邻接房间 A 和 房间 B，表示墙段在此结束
                        if region_id_map[ax, ay] != rid or region_id_map[bx, by] != encountered_room:
                            break
                        # 墙段在该方向上继续，记录像素并标记访问
                        wall_pixels_segment.append((curx, cury))
                        visited_wall[curx, cury] = True

                # 统计该墙段的像素数和记录连接信息
                segment_pixels = set(wall_pixels_segment)  # 去重获取该墙段所有独立像素
                segment_length = len(segment_pixels)
                # 初始化或更新 edges 字典中房间对的墙连接信息
                if pair not in edges:
                    edges[pair] = {
                        'connection_types': {'wall'},
                        'num_wall': 1,
                        'area_wall': segment_length,
                        'num_door_window': 0,
                        'area_door_window': 0
                    }
                else:
                    # 保留已有的连接类型，新增 'wall'
                    edges[pair]['connection_types'].add('wall')
                    # 增加墙段数量和像素面积
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += segment_length

    return edges
