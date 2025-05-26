import numpy as np
from scipy.ndimage import binary_erosion, label, binary_dilation


def get_boundary_mask(region_id_map, rid):
    """
    返回房间 `rid` 的边界像素布尔掩膜。
    边界像素定义：属于房间rid且至少有一个直接相邻（上、下、左或右）像素不属于该房间的像素。
    """
    room_mask = (region_id_map == rid)
    if not room_mask.any():
        # 若房间ID不存在于地图中，返回全False掩膜
        return np.zeros_like(region_id_map, dtype=bool)
    # 定义4-邻接结构元素（包括中心像素）
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    # 通过腐蚀获取房间内部像素，border_value=0 将房间外视为空
    interior = binary_erosion(room_mask, structure=structure, border_value=0)
    # 房间边界像素 = 房间mask减去其内部像素
    boundary_mask = room_mask & ~interior
    return boundary_mask


def detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array, debug=True):
    """
    检测室内房间之间的邻接关系（门、窗、墙连接）。
    返回一个字典 `edges`，键为(room1, room2)元组，值为包含以下字段的字典：
      - 'connection_types': 包含 {'door', 'window', 'wall'} 的集合，指示房间间存在的连接类型。
      - 'num_door_window': 房间间门/窗洞开口的数量。
      - 'area_door_window': 上述门/窗开口的总像素面积（像素数）。
      - 'num_wall': 房间间墙体连接段的数量。
      - 'area_wall': 上述墙体段的总像素数（面积近似值）。
    参数：
      - region_id_map: 房间分区ID的二维数组，0表示背景/非房间。
      - wall_array: 墙体像素的二值数组（1表示墙体，0表示非墙体）。
      - icon_array: 图标数组，用于标识门窗等开口区域（约定1为门，2为窗）。
      - wall_label_array: 墙体类别/分段标签数组，用于辅助判断外墙等排除情况。
      - debug: 是否启用调试模式。若为True，将输出调试图像'debug_internal_wall_mask.png'显示内部墙体区域。
    """
    edges = {}

    def ensure_edge_entry(a, b):
        """确保字典中存在房间对(a,b)的记录条目（无向，无序）。"""
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

    # 定义4-邻接结构（用于连通域检测和膨胀操作）
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)

    # **门和窗的连接检测（保持与v0逻辑一致）**
    for icon_val, icon_type in [(1, 'door'), (2, 'window')]:
        icon_mask = (icon_array == icon_val)
        labeled_icons, num_icons = label(icon_mask, structure=structure)
        for lbl in range(1, num_icons + 1):
            comp_mask = (labeled_icons == lbl)
            if not comp_mask.any():
                continue
            # 膨胀图标区域一个像素，以找到其相邻的房间区域
            dilated = binary_dilation(comp_mask, structure=structure, border_value=0)
            neighbor_area = dilated & ~comp_mask  # 图标区域膨胀后减去自身，得到其周围邻接区域
            neighbor_ids = np.unique(region_id_map[neighbor_area])
            neighbor_ids = neighbor_ids[neighbor_ids > 0]  # 过滤掉背景（0）
            if neighbor_ids.size == 2:
                # 图标正好连接两个不同的房间
                a, b = int(neighbor_ids[0]), int(neighbor_ids[1])
                key = ensure_edge_entry(a, b)
                edges[key]['connection_types'].add(icon_type)  # 添加连接类型 'door' 或 'window'
                edges[key]['num_door_window'] += 1
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **墙体连接检测**
    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]  # 排除背景ID 0
    visited_wall = np.zeros_like(wall_array, dtype=bool)  # 标记墙体像素是否已处理，避免重复计数
    internal_wall_mask = np.zeros_like(wall_array, dtype=bool)  # 调试掩膜，用于标记有效内部墙体像素

    # 定义需要排除的墙体类别索引（外墙、背景或特殊标记等）
    exclude_indices = {0, 1, 8, 50}

    # 遍历每个房间的边界，尝试从边界像素出发扫描墙体
    for rid in region_ids:
        # 提取当前房间rid的边界像素掩膜
        boundary_mask = get_boundary_mask(region_id_map, rid)
        boundary_coords = np.transpose(np.nonzero(boundary_mask))  # 边界像素坐标列表

        # 遍历该房间的每个边界像素
        for (x, y) in boundary_coords:
            # 检查四个正交方向的相邻像素
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # 跳过越界或非墙体的邻居
                if nx < 0 or nx >= region_id_map.shape[0] or ny < 0 or ny >= region_id_map.shape[1]:
                    continue
                if wall_array[nx, ny] != 1:
                    continue
                # 如果该墙体邻居像素已被处理过，则跳过避免重复
                if visited_wall[nx, ny]:
                    continue

                # 邻居是未访问过的墙体像素，从此点沿当前方向进入墙体扫描
                wall_pixels_segment = []  # 存储当前墙段的所有墙体像素
                encountered_room = None  # 在穿透墙体过程中遇到的另一房间ID
                last_wall_pixel = None  # 穿透扫描结束时最后一个墙体像素的坐标（用于墙段延伸）

                t = 1
                # 沿(dx, dy)方向逐像素前进穿过墙体
                while True:
                    cx = x + dx * t
                    cy = y + dy * t
                    # 超出平面边界，停止扫描
                    if cx < 0 or cx >= region_id_map.shape[0] or cy < 0 or cy >= region_id_map.shape[1]:
                        break
                    # 若仍在墙体内，则记录该墙体像素并继续前进
                    if wall_array[cx, cy] == 1:
                        wall_pixels_segment.append((cx, cy))
                        t += 1
                        continue
                    # 遇到非墙体像素，停止扫描
                    else:
                        rid2 = region_id_map[cx, cy]
                        # 如果碰到的是另一个房间的像素（rid2非0且不等于当前房间rid）
                        if rid2 != 0 and rid2 != rid:
                            encountered_room = int(rid2)
                            # 记录扫描停止前的最后一个墙体像素（即(cx,cy)前一个仍在墙内的像素）
                            last_wall_pixel = (cx - dx, cy - dy)
                        # 无论是否找到另一个房间，都结束当前方向的扫描
                        break

                # 若没有直通另一房间（encountered_room为空），则不是有效的房间间墙连接
                if encountered_room is None:
                    continue

                # 确定房间对(pair)的键，以房间ID小的一方在前（无向）
                id1, id2 = rid, encountered_room
                pair = (id1, id2) if id1 < id2 else (id2, id1)

                # 将初始扫描经过的墙体像素标记为已访问
                for (wx, wy) in wall_pixels_segment:
                    visited_wall[wx, wy] = True

                # 根据扫描方向确定墙段走向和房间所在的侧向，用于沿墙段长度方向扩展
                if dx == 0 and dy == 1:  # 从西向东扫描，墙段走向为垂直（上下方向扩展）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, -1), (0, 1)  # A房间在墙的西侧（左侧），B房间在墙的东侧（右侧）
                elif dx == 0 and dy == -1:  # 从东向西扫描，墙段走向为垂直（上下方向扩展）
                    orient_dirs = [(1, 0), (-1, 0)]
                    sideA, sideB = (0, 1), (0, -1)  # A房间在墙的东侧，B房间在墙的西侧
                elif dx == 1 and dy == 0:  # 从北向南扫描，墙段走向为水平（左右方向扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (-1, 0), (1, 0)  # A房间在墙的北侧，上侧；B房间在墙的南侧，下侧
                elif dx == -1 and dy == 0:  # 从南向北扫描，墙段走向为水平（左右方向扩展）
                    orient_dirs = [(0, 1), (0, -1)]
                    sideA, sideB = (1, 0), (-1, 0)  # A房间在墙的南侧，B房间在墙的北侧
                else:
                    orient_dirs = []

                # 使用扫描穿透结束位置的最后墙体像素作为基准点
                base_x, base_y = last_wall_pixel if last_wall_pixel is not None else wall_pixels_segment[-1]

                # 分别向墙段的两端（orient_dirs方向）延伸，收集整个连续墙段的墙体像素
                for (odx, ody) in orient_dirs:
                    curx, cury = base_x, base_y
                    while True:
                        curx += odx
                        cury += ody
                        # 越界则停止延伸
                        if curx < 0 or curx >= region_id_map.shape[0] or cury < 0 or cury >= region_id_map.shape[1]:
                            break
                        # 非墙体则停止延伸
                        if wall_array[curx, cury] != 1:
                            break
                        # 如果该墙体像素已被记录过，表示此墙段已处理过，停止延伸避免重复
                        if visited_wall[curx, cury]:
                            break
                        # 检查该墙体像素两侧是否仍分别邻接房间rid（A）和encountered_room（B）
                        ax, ay = curx + sideA[0], cury + sideA[1]  # 朝A房间侧相邻的位置
                        bx, by = curx + sideB[0], cury + sideB[1]  # 朝B房间侧相邻的位置
                        # 若任一侧越界，停止（视作不再邻接对应房间）
                        if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                            break
                        if bx < 0 or bx >= region_id_map.shape[0] or by < 0 or by >= region_id_map.shape[1]:
                            break
                        # 若当前墙像素两侧不再同时邻接房间A和房间B，则墙段在此方向终止
                        if region_id_map[ax, ay] != rid or region_id_map[bx, by] != encountered_room:
                            break
                        # 满足条件，记录该墙体像素属于同一墙段并标记已访问
                        wall_pixels_segment.append((curx, cury))
                        visited_wall[curx, cury] = True

                # 获取该墙段所有墙体像素的集合（去重）
                segment_pixels = set(wall_pixels_segment)
                segment_length = len(segment_pixels)

                # **外墙及异常情况排除判断**
                exclude_segment = False
                for (wx, wy) in segment_pixels:
                    # 检查墙段像素周围是否有属于排除类别的标签
                    for (adx, ady) in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = wx + adx, wy + ady
                        if nx < 0 or nx >= wall_label_array.shape[0] or ny < 0 or ny >= wall_label_array.shape[1]:
                            continue
                        if wall_label_array[nx, ny] in exclude_indices:
                            exclude_segment = True
                            break
                    if exclude_segment:
                        break
                if exclude_segment:
                    # 若该墙段与外墙/背景等相邻，不计入房间连接
                    continue

                # 将该墙段的像素标记到调试掩膜中
                for (wx, wy) in segment_pixels:
                    internal_wall_mask[wx, wy] = True

                # 更新edges字典中房间对的墙连接信息
                key = ensure_edge_entry(rid, encountered_room)
                edges[key]['connection_types'].add('wall')
                edges[key]['num_wall'] += 1
                edges[key]['area_wall'] += segment_length

    # 输出调试图像：显示所有判定为内部墙体的像素区域（白色）
    if debug:
        try:
            import imageio
            # 将内部墙掩膜转换为0-255的图像并保存
            imageio.imwrite('debug_internal_wall_mask.png', (internal_wall_mask.astype(np.uint8) * 255))
        except ImportError:
            # 若没有imageio，可改用其他库保存图像，例如opencv或PIL
            pass

    return edges
