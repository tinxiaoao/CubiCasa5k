import cv2
import numpy as np


def detect_adjacency(region_id_map, wall_array, icon_array):
    """
    检测室内房间之间的相邻关系，包括通过墙段、门窗和无墙开口相邻的情况。
    参数:
        region_id_map: 2D数组，每个像素值为对应房间的ID（0表示非房间区域，如墙或背景）。
        wall_array: 2D二值数组，表示墙体像素位置（1为墙体，0为非墙）。
        icon_array: 2D数组，表示图例（门、窗等）的位置及类别，用不同的数值编码。
        wall_label_array: 2D数组，标识墙体的连通区域（每段墙有唯一标签）。本版本中不直接使用该参数。
    返回:
        edges: 列表，包含若干字典，每个字典表示一条相邻边，格式:
            {
                'roomA': id1,
                'roomB': id2,
                'type': 'wall'/'door'/'window'/'opening',
                'length': L,
                'width': W
            }
    """
    edges = []

    # 1. 门窗连接检测（保留原逻辑）
    # 使用 cv2.connectedComponents 提取所有门/窗图例的连通区域
    icon_mask = np.zeros(icon_array.shape, dtype=np.uint8)
    icon_mask[icon_array > 0] = 1  # 将所有图例像素设为1（二值化）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(icon_mask, connectivity=8)
    # 遍历每个连通的门/窗区域
    for label in range(1, num_labels):
        comp_mask = (labels == label)
        # 查找该连通组件相邻的房间ID（检查其4邻域像素的房间编号）
        room_neighbors = set()
        comp_coords = np.argwhere(comp_mask)
        for (pi, pj) in comp_coords:
            if pi > 0:
                rid = region_id_map[pi - 1, pj]
                if rid > 0:
                    room_neighbors.add(int(rid))
            if pi < region_id_map.shape[0] - 1:
                rid = region_id_map[pi + 1, pj]
                if rid > 0:
                    room_neighbors.add(int(rid))
            if pj > 0:
                rid = region_id_map[pi, pj - 1]
                if rid > 0:
                    room_neighbors.add(int(rid))
            if pj < region_id_map.shape[1] - 1:
                rid = region_id_map[pi, pj + 1]
                if rid > 0:
                    room_neighbors.add(int(rid))
        # 仅当邻接恰好两个不同房间时，确定为有效的门/窗连接
        if len(room_neighbors) == 2:
            roomA, roomB = sorted(list(room_neighbors))
            # 判断图例类型：根据 icon_array 中该区域像素值（假定门=3, 窗=4 等）
            comp_values = np.unique(icon_array[comp_mask])
            comp_values = comp_values[comp_values != 0]  # 排除背景0
            comp_type = None
            if comp_values.size > 0:
                comp_class = int(comp_values[0])
                if comp_class == 3:
                    comp_type = 'door'
                elif comp_class == 4:
                    comp_type = 'window'
            # 如果类型无法判断则跳过（通常不会发生）
            if comp_type is None:
                continue
            # 计算门/窗的长度和厚度：采用组件外接矩形尺寸
            x, y, w, h, area = stats[label]
            length = int(max(w, h))
            width = int(min(w, h))
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': comp_type,
                'length': length,
                'width': width
            })

    # 2. 墙段切割与归属逻辑
    # 使用墙体像素的邻域关系，将墙划分给相邻的房间对
    wall_pair_pixels = {}  # 字典: (roomA, roomB) -> 墙体像素列表
    wall_coords = np.argwhere(wall_array > 0)
    for (i, j) in wall_coords:
        # 获取该墙像素的 8 邻域内相邻的房间ID集合
        neighbor_ids = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < region_id_map.shape[0] and 0 <= nj < region_id_map.shape[1]:
                    rid = region_id_map[ni, nj]
                    if rid > 0:
                        neighbor_ids.add(int(rid))
        # 若恰有两个不同房间与该墙像素相邻，则记为该房间对的隔墙像素
        if len(neighbor_ids) == 2:
            roomA, roomB = sorted(list(neighbor_ids))
            pair_key = (roomA, roomB)
            if pair_key not in wall_pair_pixels:
                wall_pair_pixels[pair_key] = []
            wall_pair_pixels[pair_key].append((i, j))
        else:
            # 特殊情况: 邻接房间数≠2（如0或>2），不参与房间连接标记，避免歧义
            # 例如墙体交汇的三岔点处，一个墙像素邻接多个房间，跳过以防止错误归属
            continue

    # 将每对房间的墙体像素按连通性聚类，得到各连续墙段
    for pair, pixels in wall_pair_pixels.items():
        roomA, roomB = pair
        # 构建该房间对的墙像素掩膜
        temp_mask = np.zeros(wall_array.shape, dtype=np.uint8)
        for (pi, pj) in pixels:
            temp_mask[pi, pj] = 1
        # 连通组件分析，每个连通区域对应一段连续墙体
        num_comps, comp_labels, comp_stats, comp_centroids = cv2.connectedComponentsWithStats(temp_mask, connectivity=8)
        for comp_label in range(1, num_comps):
            x, y, w, h, area = comp_stats[comp_label]
            length = int(max(w, h))
            width = int(min(w, h))
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': 'wall',
                'length': length,
                'width': width
            })
            # 注意: 若门窗在墙上开洞导致墙段中断，同一房间对可能产生多段墙体边
            # 我们不合并这些段，每段墙体都作为独立的相邻边输出

    # 3. 开敞相邻的房间检测（无墙直接接触）
    # 查找 region_id_map 中直接相邻且没有墙隔开的房间对
    open_pairs = {}  # 字典: (roomA, roomB) -> 相邻像素接触信息列表
    rows, cols = region_id_map.shape
    for i in range(rows):
        for j in range(cols):
            current_id = int(region_id_map[i, j])
            if current_id <= 0:
                continue
            # 检查右侧相邻像素
            if j < cols - 1:
                right_id = int(region_id_map[i, j + 1])
                if right_id > 0 and right_id != current_id:
                    # 如果两不同房间像素直接相邻，且对应位置没有墙
                    if wall_array[i, j] == 0 and wall_array[i, j + 1] == 0:
                        roomA, roomB = sorted([current_id, right_id])
                        if (roomA, roomB) not in open_pairs:
                            open_pairs[(roomA, roomB)] = []
                        open_pairs[(roomA, roomB)].append(('vertical', i, j))
            # 检查下方相邻像素
            if i < rows - 1:
                down_id = int(region_id_map[i + 1, j])
                if down_id > 0 and down_id != current_id:
                    if wall_array[i, j] == 0 and wall_array[i + 1, j] == 0:
                        roomA, roomB = sorted([current_id, down_id])
                        if (roomA, roomB) not in open_pairs:
                            open_pairs[(roomA, roomB)] = []
                        open_pairs[(roomA, roomB)].append(('horizontal', i, j))

    # 将每对房间的开口相邻像素分组成连续段
    for pair, contacts in open_pairs.items():
        roomA, roomB = pair
        # 用字典分别收集垂直方向和水平方向的相邻位置
        vert_positions = {}  # 键: 边界列索引j, 值: 连续接触的行索引列表
        horiz_positions = {}  # 键: 边界行索引i, 值: 连续接触的列索引列表
        for (orientation, ii, jj) in contacts:
            if orientation == 'vertical':
                # 垂直方向接触: 说明 (ii,jj) 与 (ii,jj+1) 相邻
                # 以列jj为边界位置依据，将行ii加入该列的列表
                if jj not in vert_positions:
                    vert_positions[jj] = []
                vert_positions[jj].append(ii)
            else:  # 'horizontal'
                # 水平方向接触: 说明 (ii,jj) 与 (ii+1,jj) 相邻
                # 以行ii为边界位置依据，将列jj加入该行的列表
                if ii not in horiz_positions:
                    horiz_positions[ii] = []
                horiz_positions[ii].append(jj)
        # 处理垂直方向开口相邻段（沿竖直方向连续的相邻像素）
        for jj, i_list in vert_positions.items():
            i_list.sort()
            segment_start = i_list[0]
            prev_i = i_list[0]
            for idx in range(1, len(i_list)):
                if i_list[idx] != prev_i + 1:
                    # 出现不连续，中断开此段
                    segment_end = prev_i
                    length = segment_end - segment_start + 1
                    edges.append({
                        'roomA': roomA,
                        'roomB': roomB,
                        'type': 'opening',
                        'length': int(length),
                        'width': 0
                    })
                    segment_start = i_list[idx]  # 新段起点
                prev_i = i_list[idx]
            # 处理最后一段
            segment_end = prev_i
            length = segment_end - segment_start + 1
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': 'opening',
                'length': int(length),
                'width': 0
            })
        # 处理水平方向开口相邻段（沿水平方向连续的相邻像素）
        for ii, j_list in horiz_positions.items():
            j_list.sort()
            segment_start = j_list[0]
            prev_j = j_list[0]
            for idx in range(1, len(j_list)):
                if j_list[idx] != prev_j + 1:
                    segment_end = prev_j
                    length = segment_end - segment_start + 1
                    edges.append({
                        'roomA': roomA,
                        'roomB': roomB,
                        'type': 'opening',
                        'length': int(length),
                        'width': 0
                    })
                    segment_start = j_list[idx]
                prev_j = j_list[idx]
            segment_end = prev_j
            length = segment_end - segment_start + 1
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': 'opening',
                'length': int(length),
                'width': 0
            })

    return edges
