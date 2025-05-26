import numpy as np
from scipy import ndimage


def detect_adjacency(region_id_map: np.ndarray, wall_array: np.ndarray, icon_array: np.ndarray):
    """
    根据房间区域图 (region_id_map)、墙体二值图 (wall_array) 和门窗二值图 (icon_array)，
    检测房间之间的邻接关系，包括通过墙体连接和通过门窗（门/窗）连接。

    返回值:
        edges (dict): 邻接关系字典。
            键为 (id1, id2) 的房间对元组（id1 < id2），
            值为包含连接信息的字典，包括:
                'connection_types': set，包含 'wall'、'door'、'window' 或 'door/window'（至少包含其一）表示连接类型；
                'num_wall': int，墙体连接段数量（计数）；
                'area_wall': int，墙体连接的像素总数；
                'num_door_window': int，门窗连接数量（门和窗合计）；
                'area_door_window': int，门窗连接的像素总数。
    """
    edges = {}
    # 图像尺寸
    H, W = region_id_map.shape
    # 要排除的区域索引（背景、Outdoor、栏杆等），这些不计入房间邻接
    exclude_indices = {0, 1, 8, 50}

    # 1. 门窗连接检测逻辑（保持与 v0 版本一致）
    # 遍历门和窗两类 icon，进行连通域分析
    for icon_val, conn_label in [(1, 'door'), (2, 'window')]:
        # 二值掩膜：当前类型的 icon（1=门 或 2=窗）
        icon_mask = (icon_array == icon_val).astype(np.uint8)
        if icon_mask.any():
            # 连通域标记（使用8连通保证对角相连的像素属于同一连通块）
            labeled_array, num_features = ndimage.label(icon_mask, structure=np.ones((3, 3), dtype=int))
            for lbl in range(1, num_features + 1):
                # 提取该连通块的所有像素坐标
                coords = np.argwhere(labeled_array == lbl)
                neighbor_ids = set()
                skip_segment = False
                # 检查该门/窗连通块周围相邻的房间区域
                for (x, y) in coords:
                    # 四邻接像素坐标偏移量 (上、下、左、右)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if nx < 0 or nx >= H or ny < 0 or ny >= W:
                            # 与图像边界相邻，视为外部区域
                            skip_segment = True
                            break
                        region_val = region_id_map[nx, ny]
                        if region_val > 0:
                            neighbor_ids.add(region_val)
                        elif region_val in exclude_indices or region_val == 0:
                            # 若邻接像素属于排除类别（背景/Outdoor等）
                            # 进一步判断该邻接像素是否不是墙或其他门窗（即真正外部区域）
                            if wall_array[nx, ny] == 0 and icon_array[nx, ny] == 0:
                                skip_segment = True
                                break
                            # 如果邻居是墙体或另一个门窗像素，则忽略（不计为房间）
                    if skip_segment:
                        break
                # 如果该门/窗段无效（相邻区域少于2，或接触外部），则跳过
                if skip_segment or len(neighbor_ids) < 2:
                    continue
                # 获取相邻房间ID对（一般应正好2个）
                neighbor_ids = sorted(neighbor_ids)
                for i in range(len(neighbor_ids)):
                    for j in range(i + 1, len(neighbor_ids)):
                        id1, id2 = neighbor_ids[i], neighbor_ids[j]
                        if id1 == id2:
                            continue
                        pair = (id1, id2) if id1 < id2 else (id2, id1)
                        if pair not in edges:
                            edges[pair] = {
                                'connection_types': set(),
                                'num_wall': 0,
                                'area_wall': 0,
                                'num_door_window': 0,
                                'area_door_window': 0
                            }
                        # 标记连接类型，累加门窗连接数量和面积
                        edges[pair]['connection_types'].add(conn_label)
                        edges[pair]['num_door_window'] += 1
                        edges[pair]['area_door_window'] += coords.shape[0]
    # 2. 墙体连接检测逻辑
    if wall_array.any():
        # 对墙体像素进行连通域分析（8连通，将相连墙段视为一整段）
        labeled_walls, num_wall_components = ndimage.label(wall_array, structure=np.ones((3, 3), dtype=int))
        for lbl in range(1, num_wall_components + 1):
            coords = np.argwhere(labeled_walls == lbl)
            neighbor_ids = set()
            skip_segment = False
            for (x, y) in coords:
                # 检查每个墙体像素周围的房间区域像素（4连通方向）
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= H or ny < 0 or ny >= W:
                        # 墙段接触图像边界，视为外墙，整段跳过
                        skip_segment = True
                        break
                    region_val = region_id_map[nx, ny]
                    if region_val > 0:
                        neighbor_ids.add(region_val)
                    elif region_val in exclude_indices or region_val == 0:
                        # 墙段邻接排除区域（背景、室外、栏杆等），跳过该墙段
                        if wall_array[nx, ny] == 0 and icon_array[nx, ny] == 0:
                            skip_segment = True
                            break
                        # 邻居若是墙体或门窗，同样不计入房间集合
                if skip_segment:
                    break
            # 如果该墙段接触外部或邻接房间数不足2，则跳过
            if skip_segment or len(neighbor_ids) < 2:
                continue
            # 识别该墙段连接的所有房间对
            neighbor_ids = sorted(neighbor_ids)
            for i in range(len(neighbor_ids)):
                for j in range(i + 1, len(neighbor_ids)):
                    id1, id2 = neighbor_ids[i], neighbor_ids[j]
                    if id1 == id2:
                        continue
                    pair = (id1, id2) if id1 < id2 else (id2, id1)
                    if pair not in edges:
                        edges[pair] = {
                            'connection_types': set(),
                            'num_wall': 0,
                            'area_wall': 0,
                            'num_door_window': 0,
                            'area_door_window': 0
                        }
                    # 增加墙体连接信息（每段墙体连通块计为一次连接）
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += coords.shape[0]
    return edges
