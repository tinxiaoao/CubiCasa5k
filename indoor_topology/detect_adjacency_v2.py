import numpy as np
from scipy.ndimage import binary_erosion, label, binary_dilation

def get_boundary_mask(region_id_map, rid):
    """
    返回房间 `rid` 的边界像素布尔掩膜。
    边界像素定义：属于房间 rid 且至少有一个直接相邻（上、下、左或右）像素不属于该房间的像素。
    """
    room_mask = (region_id_map == rid)
    if not room_mask.any():
        # 若房间 ID 不存在于地图中，返回全 False 掩膜
        return np.zeros_like(region_id_map, dtype=bool)
    # 定义 4-邻接结构元素（包括中心像素）
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    # 通过腐蚀获取房间内部像素（border_value=0 将房间外视为空）
    interior = binary_erosion(room_mask, structure=structure, border_value=0)
    # 房间边界像素 = 房间 mask 减去其内部像素
    boundary_mask = room_mask & ~interior
    return boundary_mask

def detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array, debug=True):
    """
    检测室内房间之间的邻接关系（门、窗、墙连接）。
    返回一个字典 `edges`，键为 (room1, room2) 元组，值为包含以下字段的字典：
      - 'connection_types': 包含 {'door', 'window', 'wall'} 的集合，指示房间间存在的连接类型。
      - 'num_door_window': 房间间门/窗洞开口的数量。
      - 'area_door_window': 上述门/窗开口的总像素面积（像素数）。
      - 'num_wall': 房间间墙体连接段的数量。
      - 'area_wall': 上述墙体段的总像素数（面积近似值）。
    参数：
      - region_id_map: 房间分区 ID 的二维数组，0 表示背景/非房间。
      - wall_array: 墙体像素的二值数组（1 表示墙体，0 表示非墙体）。
      - icon_array: 图标数组，用于标识门窗等开口区域（约定 1 为门，2 为窗）。
      - wall_label_array: 墙体类别/分段标签数组，用于辅助判断外墙等需要排除的情况。
      - debug: 调试模式标志（当前实现不输出调试图像）。
    """
    edges = {}

    # 辅助函数：确保初始化房间对字典条目
    def ensure_edge_entry(a, b):
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

    # 4-邻接结构元素（十字形）
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
            # 找到与该门/窗区域相邻的房间 ID 集合
            dilated = binary_dilation(comp_mask, structure=structure, border_value=0)
            neighbor_area = dilated & ~comp_mask
            neighbor_ids = np.unique(region_id_map[neighbor_area])
            neighbor_ids = neighbor_ids[neighbor_ids > 0]  # 排除背景 0
            if neighbor_ids.size == 2:
                a = int(neighbor_ids[0])
                b = int(neighbor_ids[1])
                key = ensure_edge_entry(a, b)
                edges[key]['connection_types'].add(icon_type)
                edges[key]['num_door_window'] += 1
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **Wall connections**
    # 提取所有房间 ID（排除背景 0）
    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]

    # 墙体访问标记数组，避免重复扫描
    visited_wall = np.zeros_like(wall_array, dtype=bool)
    # 定义需要排除的墙体标签索引（如外墙、背景等）
    exclude_indices = {0, 1, 8, 50}
    # 字典：记录每个墙段已连接的房间对及其最小连接面积
    segment_label_map = {}

    # 遍历每个房间的边界像素，沿墙体穿透查找相邻房间
    for rid_val in region_ids:
        rid = int(rid_val)
        boundary_mask = get_boundary_mask(region_id_map, rid)
        boundary_coords = np.transpose(np.nonzero(boundary_mask))
        for x, y in boundary_coords:
            # 检查四个正交方向的邻居像素
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # 跳过越界或非墙体的邻居
                if nx < 0 or nx >= region_id_map.shape[0] or ny < 0 or ny >= region_id_map.shape[1]:
                    continue
                if wall_array[nx, ny] != 1:
                    continue
                # 邻居是墙体且未被访问，开始沿该方向穿墙扫描
                if visited_wall[nx, ny]:
                    continue
                t = 1
                wall_pixels_segment = []
                encountered_room = None
                last_wall_pixel = None
                # 沿当前方向逐像素穿过墙体
                while True:
                    cx = x + dx * t
                    cy = y + dy * t
                    # 越界则停止扫描
                    if cx < 0 or cx >= region_id_map.shape[0] or cy < 0 or cy >= region_id_map.shape[1]:
                        break
                    if wall_array[cx, cy] == 1:
                        # 仍在墙体内部，记录像素后继续前进
                        wall_pixels_segment.append((cx, cy))
                        t += 1
                        continue
                    else:
                        # 碰到非墙体像素，结束扫描
                        rid2 = region_id_map[cx, cy]
                        if rid2 != 0 and rid2 != rid:
                            # 穿墙另一侧遇到另一房间
                            encountered_room = int(rid2)
                            last_wall_pixel = (cx - dx, cy - dy)
                        break
                # 未穿透到另一房间，则不是房间间墙连接
                if encountered_room is None:
                    continue
                # 确定房间对 (rid, encountered_room)，按大小排序作为无向键
                id1, id2 = rid, encountered_room
                pair = (id1, id2) if id1 < id2 else (id2, id1)

                # 标记扫描经过的墙体像素为已访问
                for wx, wy in wall_pixels_segment:
                    visited_wall[wx, wy] = True

                # 根据初始扫描方向确定墙段走向及两侧房间，用于延伸收集整段墙体
                if dx == 0 and dy == 1:    # 向东扫描，墙段走向垂直（上下延伸）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, -1), (0, 1)    # 房间 A 在墙西侧，B 在墙东侧
                elif dx == 0 and dy == -1: # 向西扫描，墙段走向垂直（上下延伸）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, 1), (0, -1)    # 房间 A 在墙东侧，B 在墙西侧
                elif dx == 1 and dy == 0:  # 向南扫描，墙段走向水平（左右延伸）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (-1, 0), (1, 0)    # 房间 A 在墙北侧，B 在墙南侧
                elif dx == -1 and dy == 0: # 向北扫描，墙段走向水平（左右延伸）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (1, 0), (-1, 0)    # 房间 A 在墙南侧，B 在墙北侧
                else:
                    orient_dirs = []

                # 以穿墙终点的最后一个墙体像素为基准点
                base_x, base_y = last_wall_pixel if last_wall_pixel is not None else wall_pixels_segment[-1]

                # 分别向墙段两端延伸，收集整个连续墙段的墙体像素
                for odx, ody in orient_dirs:
                    curx, cury = base_x, base_y
                    while True:
                        curx += odx
                        cury += ody
                        # 越界则停止延伸
                        if curx < 0 or curx >= region_id_map.shape[0] or cury < 0 or cury >= region_id_map.shape[1]:
                            break
                        if wall_array[curx, cury] != 1:
                            break
                        if visited_wall[curx, cury]:
                            break
                        ax = curx + sideA[0]; ay = cury + sideA[1]
                        bx = curx + sideB[0]; by = cury + sideB[1]
                        if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                            break
                        if bx < 0 or bx >= region_id_map.shape[0] or by < 0 or by >= region_id_map.shape[1]:
                            break
                        if region_id_map[ax, ay] != rid or region_id_map[bx, by] != encountered_room:
                            break
                        wall_pixels_segment.append((curx, cury))
                        visited_wall[curx, cury] = True

                # 计算整段墙体的像素数量（面积近似值）
                segment_pixels = set(wall_pixels_segment)
                segment_length = len(segment_pixels)

                # 排除与外墙或异常标签相邻的墙段
                exclude_segment = False
                for (wx, wy) in segment_pixels:
                    for (adx, ady) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx2, ny2 = wx + adx, wy + ady
                        if nx2 < 0 or nx2 >= wall_label_array.shape[0] or ny2 < 0 or ny2 >= wall_label_array.shape[1]:
                            continue
                        if wall_label_array[nx2, ny2] in exclude_indices:
                            exclude_segment = True
                            break
                    if exclude_segment:
                        break
                if exclude_segment:
                    continue

                # 每段墙体仅保留一个房间对连接（若共享则保留面积最小者）
                seg_label = wall_label_array[base_x, base_y]
                if seg_label not in segment_label_map:
                    # 首次发现该墙段，记录房间对及墙段面积
                    segment_label_map[seg_label] = (pair, segment_length)
                    key = ensure_edge_entry(id1, id2)
                    edges[key]['connection_types'].add('wall')
                    edges[key]['num_wall'] += 1
                    edges[key]['area_wall'] += segment_length
                else:
                    prev_pair, prev_len = segment_label_map[seg_label]
                    if segment_length < prev_len:
                        # 发现更小面积的房间对连接，替换之前的记录
                        edges[prev_pair]['num_wall'] -= 1
                        edges[prev_pair]['area_wall'] -= prev_len
                        if edges[prev_pair]['num_wall'] == 0:
                            edges[prev_pair]['connection_types'].discard('wall')
                            if len(edges[prev_pair]['connection_types']) == 0:
                                edges.pop(prev_pair, None)
                        segment_label_map[seg_label] = (pair, segment_length)
                        key = ensure_edge_entry(id1, id2)
                        edges[key]['connection_types'].add('wall')
                        edges[key]['num_wall'] += 1
                        edges[key]['area_wall'] += segment_length
                    else:
                        # 丢弃较大面积的重复墙连接
                        continue

    # 返回 edges 字典结果（不输出调试图像）
    return edges
