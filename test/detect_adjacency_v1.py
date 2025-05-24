import numpy as np
from scipy import ndimage


def detect_adjacency(region_id_map, wall_array, icon_array):
    """
    检测房间之间通过门窗或墙相连的关系，返回房间连接信息字典。
    输入:
        region_id_map: numpy.ndarray，房间区域标识矩阵，每个像素的值为所属房间的ID，0表示非房间区域(墙体或背景)。
        wall_array: numpy.ndarray，二值矩阵，表示墙体区域(1为墙体像素，0为非墙)。
        icon_array: numpy.ndarray，矩阵，表示门窗图标(1代表门，2代表窗，0为无图标)。
    输出:
        edges: dict，房间对 -> 连接信息的字典。
            键: (room_id1, room_id2) 元组，使用房间ID较小的在前顺序。
            值: 包含以下键的字典：
                - connection_types: set，{'door', 'window', 'wall'}的子集，表示存在的连接类型。
                - num_door_window: int，门和窗连接的数量之和。
                - area_door_window: int，门和窗连接的总面积(以像素计)。
                - num_wall: int，墙连接段数量。
                - area_wall: int，墙连接段的总面积(以像素计)。
    """
    edges = {}

    # 定义4-邻域结构元素，用于连通区域分析和膨胀
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=int)

    # 1. 门窗连接检测
    # 标记所有门(icon_array==1)的连通区域和所有窗(icon_array==2)的连通区域
    door_mask = (icon_array == 1)
    window_mask = (icon_array == 2)
    labeled_doors, num_doors = ndimage.label(door_mask, structure=structure)
    labeled_windows, num_windows = ndimage.label(window_mask, structure=structure)

    # 遍历每个门连通区域
    for label in range(1, num_doors + 1):
        comp_mask = (labeled_doors == label)
        # 膨胀门区域一圈，找到与门相邻的房间像素
        neighbor_mask = ndimage.binary_dilation(comp_mask, structure=structure) & (region_id_map > 0)
        neighbor_ids = np.unique(region_id_map[neighbor_mask])
        neighbor_ids = neighbor_ids[neighbor_ids > 0]  # 排除值0
        if neighbor_ids.size < 2:
            # 少于两个不同房间，说明该门不连接两个房间（可能一侧通向外部），跳过
            continue
        # 假定门正好连接两个房间，取出两个房间ID
        room1, room2 = int(neighbor_ids[0]), int(neighbor_ids[1])
        if room1 == room2:
            continue  # 异常情况: 两侧是同一房间
        pair = (room1, room2) if room1 < room2 else (room2, room1)
        # 初始化字典条目
        if pair not in edges:
            edges[pair] = {
                'connection_types': set(),
                'num_door_window': 0,
                'area_door_window': 0,
                'num_wall': 0,
                'area_wall': 0
            }
        # 更新房间对连接信息
        edges[pair]['connection_types'].add('door')
        edges[pair]['num_door_window'] += 1
        edges[pair]['area_door_window'] += int(comp_mask.sum())  # 门区域的像素面积累加

    # 遍历每个窗连通区域（逻辑与上述类似）
    for label in range(1, num_windows + 1):
        comp_mask = (labeled_windows == label)
        neighbor_mask = ndimage.binary_dilation(comp_mask, structure=structure) & (region_id_map > 0)
        neighbor_ids = np.unique(region_id_map[neighbor_mask])
        neighbor_ids = neighbor_ids[neighbor_ids > 0]
        if neighbor_ids.size < 2:
            # 窗户通常一侧为房间，另一侧为室外，不计入房间对连接
            continue
        room1, room2 = int(neighbor_ids[0]), int(neighbor_ids[1])
        if room1 == room2:
            continue
        pair = (room1, room2) if room1 < room2 else (room2, room1)
        if pair not in edges:
            edges[pair] = {
                'connection_types': set(),
                'num_door_window': 0,
                'area_door_window': 0,
                'num_wall': 0,
                'area_wall': 0
            }
        edges[pair]['connection_types'].add('window')
        edges[pair]['num_door_window'] += 1
        edges[pair]['area_door_window'] += int(comp_mask.sum())  # 窗区域像素面积累加

    # 2. 墙连接检测
    # 构造墙体掩膜，排除任何门窗像素，确保只在完整墙体内搜索
    wall_mask = (wall_array.astype(bool)) & (icon_array == 0)
    # 创建访问标记矩阵，避免重复遍历墙区域
    visited_wall = np.zeros(wall_mask.shape, dtype=bool)
    # 获取所有房间ID（跳过0）
    room_ids = [rid for rid in np.unique(region_id_map) if rid != 0]

    for rid in room_ids:
        # 当前房间rid的像素掩膜
        room_mask = (region_id_map == rid)
        if not room_mask.any():
            continue
        # 找出与房间相邻的墙体像素：膨胀房间区域，然后取交集在墙体内的部分
        wall_neighbors = ndimage.binary_dilation(room_mask, structure=structure) & wall_mask
        # 筛选尚未访问过的墙邻接像素作为BFS起点
        start_positions = np.transpose(np.nonzero(wall_neighbors & ~visited_wall))
        if start_positions.size == 0:
            continue  # 没有新的墙体起点，可能该房间已处理或无直接墙相邻
        # 初始化队列进行 BFS，多源同时出发
        from collections import deque
        queue = deque()
        # 距离矩阵，用于记录每个墙体像素距离当前房间边界的步数
        dist_map = -np.ones(wall_mask.shape, dtype=int)
        for x, y in start_positions:
            queue.append((x, y))
            visited_wall[x, y] = True  # 标记为已访问
            dist_map[x, y] = 1  # 距离计为1（从房间进入墙内一步）

        # BFS 遍历墙体连通区域
        while queue:
            x, y = queue.popleft()
            d = dist_map[x, y]
            # 检查当前墙像素是否邻接另一个房间
            # 如果上下左右的相邻像素属于不同的房间(other_id)，则记录墙连接
            if x > 0 and region_id_map[x - 1, y] > 0 and region_id_map[x - 1, y] != rid:
                other_id = int(region_id_map[x - 1, y])
                # 排除自身房间rid，本身region_id_map[x-1,y]>0保证是房间像素
                if other_id != rid:
                    pair = (rid, other_id) if rid < other_id else (other_id, rid)
                    if pair not in edges:
                        edges[pair] = {
                            'connection_types': set(),
                            'num_door_window': 0,
                            'area_door_window': 0,
                            'num_wall': 0,
                            'area_wall': 0
                        }
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1  # 墙连接段计数加1
                    edges[pair]['area_wall'] += d  # 累加墙连接段的路径长度
            if x < region_id_map.shape[0] - 1 and region_id_map[x + 1, y] > 0 and region_id_map[x + 1, y] != rid:
                other_id = int(region_id_map[x + 1, y])
                if other_id != rid:
                    pair = (rid, other_id) if rid < other_id else (other_id, rid)
                    if pair not in edges:
                        edges[pair] = {
                            'connection_types': set(),
                            'num_door_window': 0,
                            'area_door_window': 0,
                            'num_wall': 0,
                            'area_wall': 0
                        }
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += d
            if y > 0 and region_id_map[x, y - 1] > 0 and region_id_map[x, y - 1] != rid:
                other_id = int(region_id_map[x, y - 1])
                if other_id != rid:
                    pair = (rid, other_id) if rid < other_id else (other_id, rid)
                    if pair not in edges:
                        edges[pair] = {
                            'connection_types': set(),
                            'num_door_window': 0,
                            'area_door_window': 0,
                            'num_wall': 0,
                            'area_wall': 0
                        }
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += d
            if y < region_id_map.shape[1] - 1 and region_id_map[x, y + 1] > 0 and region_id_map[x, y + 1] != rid:
                other_id = int(region_id_map[x, y + 1])
                if other_id != rid:
                    pair = (rid, other_id) if rid < other_id else (other_id, rid)
                    if pair not in edges:
                        edges[pair] = {
                            'connection_types': set(),
                            'num_door_window': 0,
                            'area_door_window': 0,
                            'num_wall': 0,
                            'area_wall': 0
                        }
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += d

            # 在墙体内向四邻继续扩展
            if x > 0 and wall_mask[x - 1, y] and not visited_wall[x - 1, y]:
                visited_wall[x - 1, y] = True
                dist_map[x - 1, y] = d + 1
                queue.append((x - 1, y))
            if x < wall_mask.shape[0] - 1 and wall_mask[x + 1, y] and not visited_wall[x + 1, y]:
                visited_wall[x + 1, y] = True
                dist_map[x + 1, y] = d + 1
                queue.append((x + 1, y))
            if y > 0 and wall_mask[x, y - 1] and not visited_wall[x, y - 1]:
                visited_wall[x, y - 1] = True
                dist_map[x, y - 1] = d + 1
                queue.append((x, y - 1))
            if y < wall_mask.shape[1] - 1 and wall_mask[x, y + 1] and not visited_wall[x, y + 1]:
                visited_wall[x, y + 1] = True
                dist_map[x, y + 1] = d + 1
                queue.append((x, y + 1))
    return edges
