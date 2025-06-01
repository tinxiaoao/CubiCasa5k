import numpy as np
import cv2


def detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array):
    """
    基于四步墙段切割与房间连接逻辑的室内拓扑提取算法。
    输入:
        region_id_map: 2D数组，房间区域ID地图，室外区域为0
        wall_array: 2D数组，墙体像素掩码，墙体像素为1，非墙为0
        icon_array: 2D数组，门窗像素掩码，门=1，窗=2，无=0
        wall_label_array: 2D数组，墙体像素的连通组件或类别标签
    输出:
        edges: List[Dict]，房间连接边列表。每个元素包含:
            'roomA': 相邻房间A的ID (较小的ID)
            'roomB': 相邻房间B的ID (较大的ID)
            'type': 相邻类型 ('wall' 墙体, 'door' 门, 'window' 窗, 'opening' 开敞)
            'length': 墙段/门窗/开口的长度 (像素数，轴对齐边界框较长边)
            'width': 墙段/门窗/开口的宽度 (像素数，轴对齐边界框较短边)
    """
    # 确保输入为numpy数组的副本，避免修改原数据
    region_id_map = np.array(region_id_map)
    wall_array = np.array(wall_array)
    icon_array = np.array(icon_array)
    wall_label_array = np.array(wall_label_array)

    edges = []  # 最终返回的边列表

    # 1️⃣ 第一阶段：门窗连接检测
    # 提取门和窗的连通组件，分别处理
    door_mask = (icon_array == 1).astype(np.uint8)
    window_mask = (icon_array == 2).astype(np.uint8)
    # 处理门组件
    num_labels, labels = cv2.connectedComponents(door_mask, connectivity=4)
    for label in range(1, num_labels):
        # 获取该连通组件的所有像素坐标
        comp_coords = np.where(labels == label)
        if comp_coords[0].size == 0:
            continue
        comp_rows = comp_coords[0]
        comp_cols = comp_coords[1]
        # 检查该门组件像素的4邻域相邻房间
        neighbor_rooms = set()
        for (r, c) in zip(comp_rows, comp_cols):
            # 上下左右四个方向
            if r > 0:
                rid = region_id_map[r - 1, c]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if r < region_id_map.shape[0] - 1:
                rid = region_id_map[r + 1, c]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if c > 0:
                rid = region_id_map[r, c - 1]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if c < region_id_map.shape[1] - 1:
                rid = region_id_map[r, c + 1]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
        # 若恰好邻接两个不同房间，则认定为门连接
        if len(neighbor_rooms) == 2:
            roomA, roomB = sorted(list(neighbor_rooms))
            # 计算门组件的边界框尺寸
            min_r = int(comp_rows.min())
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min())
            max_c = int(comp_cols.max())
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            length = max(height, width)
            thick = min(height, width)
            edges.append({
                'roomA': roomA, 'roomB': roomB,
                'type': 'door',
                'length': int(length), 'width': int(thick)
            })

    # 处理窗组件（逻辑同上）
    num_labels, labels = cv2.connectedComponents(window_mask, connectivity=4)
    for label in range(1, num_labels):
        comp_coords = np.where(labels == label)
        if comp_coords[0].size == 0:
            continue
        comp_rows = comp_coords[0]
        comp_cols = comp_coords[1]
        neighbor_rooms = set()
        for (r, c) in zip(comp_rows, comp_cols):
            if r > 0:
                rid = region_id_map[r - 1, c]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if r < region_id_map.shape[0] - 1:
                rid = region_id_map[r + 1, c]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if c > 0:
                rid = region_id_map[r, c - 1]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
            if c < region_id_map.shape[1] - 1:
                rid = region_id_map[r, c + 1]
                if rid > 0:
                    neighbor_rooms.add(int(rid))
        if len(neighbor_rooms) == 2:
            roomA, roomB = sorted(list(neighbor_rooms))
            min_r = int(comp_rows.min())
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min())
            max_c = int(comp_cols.max())
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            length = max(height, width)
            thick = min(height, width)
            edges.append({
                'roomA': roomA, 'roomB': roomB,
                'type': 'window',
                'length': int(length), 'width': int(thick)
            })

    # 2️⃣ 第二阶段：移除外墙，只保留内部隔墙
    # 1. 提取墙体骨架（一像素宽中心线）
    # 将wall_array转换为0-255的uint8类型（二值图像）以用于细化
    wall_bin = (wall_array > 0).astype(np.uint8) * 255
    # 使用 OpenCV ximgproc 的细化函数获取骨架
    skeleton = cv2.ximgproc.thinning(wall_bin)
    # 将细化结果转换为二值0/1表示
    skeleton = (skeleton > 0).astype(np.uint8)

    # 获取图像尺寸
    h, w = skeleton.shape
    # 定义8邻域坐标偏移（包含对角方向）
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1), (0, 1),
                        (1, -1), (1, 0), (1, 1)]

    # 字典：记录每个骨架像素对应的房间对 (roomA, roomB)
    point_to_pair = {}

    # 2-3. 遍历每个骨架像素点，计算法线方向并探测两侧房间ID
    for i in range(h):
        for j in range(w):
            if skeleton[i, j] != 1:
                continue  # 跳过非骨架像素
            # 收集该像素在骨架上的邻居像素（8邻接）
            neighbors = []
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and skeleton[ni, nj] == 1:
                    neighbors.append((ni, nj))
            # 根据邻居数判断骨架像素的拓扑情况
            if len(neighbors) == 2:
                # 正常直线段中的像素：使用前后两个邻居确定切线方向
                n1 = neighbors[0]
                n2 = neighbors[1]
                # 切向量 = 邻居2坐标 - 邻居1坐标
                dr = n2[0] - n1[0]
                dc = n2[1] - n1[1]
            elif len(neighbors) == 1:
                # 墙段末端像素：只有一个邻居，假设另一侧延伸一个对称点来确定切线方向
                n1 = neighbors[0]
                dr = 2 * (i - n1[0])
                dc = 2 * (j - n1[1])
            else:
                # 对于没有邻居或超过两个邻居的情况（交汇处等），不计算法线方向
                continue

            # 计算法线方向的两个向量 (垂直于切向量)
            # 切向量(dr, dc)的法线可以取 (dc, -dr) 和 (-dc, dr)
            nr1, nc1 = dc, -dr  # 法线方向1（行方向变化nr1，列方向变化nc1）
            nr2, nc2 = -dc, dr  # 法线方向2（与法线方向1相反的另一侧）

            # 计算法线两侧探测点的坐标（沿法线方向各偏移2个像素）
            p1_i, p1_j = i + nr1, j + nc1
            p2_i, p2_j = i + nr2, j + nc2

            # 定义一个辅助函数获取region_id_map中的房间ID（越界视为室外0）
            def get_region_id(pi, pj):
                if pi < 0 or pi >= h or pj < 0 or pj >= w:
                    return 0  # 越界位置视作室外区域ID 0
                return region_id_map[pi, pj]

            id1 = get_region_id(p1_i, p1_j)
            id2 = get_region_id(p2_i, p2_j)
            # 如果法线两侧接触到不同的房间ID，则记录该骨架点对应的房间对
            if id1 != id2:
                pair = tuple(sorted((int(id1), int(id2))))
                if pair[0] == pair[1]:
                    # 若两个ID相同（例如两侧都是室外0或同一房间），则跳过
                    continue
                point_to_pair[(i, j)] = pair
                # 调试输出每个识别的骨架点与房间对（可选，注释掉以减少输出）
                # print(f"Skeleton point ({i},{j}) between room {pair[0]} and room {pair[1]}")

        # 4. 基于房间对对骨架线段进行聚类（连接相邻骨架点属于同一房间对的为一段墙体）
        visited = set()
        edges = []  # 保存结果墙段信息的列表
    for point, pair in point_to_pair.items():
        if point in visited:
            continue
            # 新的墙段簇起始
        room_pair = pair
        segment_points = []
        # 深度优先搜索（或广度优先）聚类连通的骨架像素
        stack = [point]
        visited.add(point)
        while stack:
            pi, pj = stack.pop()
            segment_points.append((pi, pj))
            # 查找相邻的骨架像素
            for di, dj in neighbor_offsets:
                ni, nj = pi + di, pj + dj
                if (ni, nj) in point_to_pair and (ni, nj) not in visited:
                    # 确认邻居与当前段属于同一房间对
                    if point_to_pair[(ni, nj)] == room_pair:
                        visited.add((ni, nj))
                        stack.append((ni, nj))
            # 计算该段墙体骨架的长度（像素数量）
            segment_length = len(segment_points)
            # 保存墙段信息，roomA和roomB对应房间ID
            roomA, roomB = room_pair
            edges.append({
                'roomA': roomA,
                'roomB': roomB,
                'type': 'wall',
                'length': segment_length,
                'width': 1
            })

        # 5. 输出结果并打印调试信息
        # 打印识别出的房间对及墙段数量等调试信息
    pair_counts = {}
    for edge in edges:
        pair = (edge['roomA'], edge['roomB'])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    print("识别出的房间对及对应墙段数量:")
    for pair, count in pair_counts.items():
        room_a, room_b = pair
        print(f"房间 {room_a} - 房间 {room_b}: 墙段数 {count}")
    print(f"总共识别出 {len(edges)} 段墙体")

    # 5️⃣ 第五阶段：处理房间接壤但无墙分隔的开敞相邻
    # 寻找在region_id_map中彼此相邻且wall_array无墙分隔的房间对
    open_pair_points = {}  # 存储每对相邻房间的开敞接触像素，用于聚类连续段
    H, W = region_id_map.shape
    for r in range(H):
        for c in range(W):
            rid = region_id_map[r, c]
            if rid <= 0:
                continue  # 非房间像素跳过
            # 检查右邻居
            if c < W - 1:
                rid2 = region_id_map[r, c + 1]
                if rid2 > 0 and rid2 != rid and wall_array[r, c + 1] == 0:
                    pair = tuple(sorted((int(rid), int(rid2))))
                    # 选择将接触点标记在房间ID较小的一侧像素上
                    if pair[0] == rid:
                        mark_r, mark_c = r, c
                    else:
                        mark_r, mark_c = r, c + 1
                    open_pair_points.setdefault(pair, []).append((mark_r, mark_c))
            # 检查下邻居
            if r < H - 1:
                rid2 = region_id_map[r + 1, c]
                if rid2 > 0 and rid2 != rid and wall_array[r + 1, c] == 0:
                    pair = tuple(sorted((int(rid), int(rid2))))
                    if pair[0] == rid:
                        mark_r, mark_c = r, c
                    else:
                        mark_r, mark_c = r + 1, c
                    open_pair_points.setdefault(pair, []).append((mark_r, mark_c))
    # 对每对房间的接壤点进行连通区域提取，将连续的开敞边界作为一段opening
    for pair, points in open_pair_points.items():
        if not points:
            continue
        # 创建掩码标记所有接触点像素（标记在前面选定的一侧房间像素上）
        mask = np.zeros_like(region_id_map, dtype=np.uint8)
        for (pr, pc) in points:
            mask[pr, pc] = 1
        # 提取连续的开敞邻接段（使用4邻接，仅通过共享边界判定连续）
        num_labels, comp_label_img = cv2.connectedComponents(mask, connectivity=4)
        for lbl in range(1, num_labels):
            comp_coords = np.where(comp_label_img == lbl)
            if comp_coords[0].size == 0:
                continue
            comp_rows = comp_coords[0]
            comp_cols = comp_coords[1]
            # 计算开口段的边界框尺寸
            min_r = int(comp_rows.min())
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min())
            max_c = int(comp_cols.max())
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            length = max(height, width)
            thick = min(height, width)
            roomA, roomB = pair
            edges.append({
                'roomA': roomA, 'roomB': roomB,
                'type': 'opening',
                'length': int(length), 'width': int(thick)
            })

    return edges
