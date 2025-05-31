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
            min_r = int(comp_rows.min());
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min());
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
            min_r = int(comp_rows.min());
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min());
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
        # ★① 先找“室外 / 阳台 / 背景”像素
        outside = np.isin(wall_label_array, [0, 1, 8, 50]).astype(np.uint8)  # 0/1:False, True→1
        H, W = outside.shape

        # ★② Flood-fill：把与图像四周连通的像素全部置 1
        mask = np.zeros((H + 2, W + 2), np.uint8)  # floodFill 需要比原图大2的 mask
        cv2.floodFill(outside, mask, seedPoint=(0, 0), newVal=1)  # 0,0 位于图外黑边

        outside_fill = (outside == 1)  # True ⇒ 室外连通域

        # ★③ 得到“室外连通域边缘” (可选，仅调试可视)
        kernel = np.ones((3, 3), np.uint8)
        outside_edge = cv2.dilate(outside_fill.astype(np.uint8), kernel, 1) & (~outside_fill)

        # ★④ 初始墙体掩码
        wall_mask = (wall_label_array == 2).astype(np.uint8)  # 1=墙, 0=其他

        # ★⑤ 与室外连通域直接接触的墙体像素
        #    （dilate 一圈，让整段墙厚首层都被打种子）
        candidate = (cv2.dilate(outside_fill.astype(np.uint8), kernel, 1) & wall_mask).astype(np.uint8)

        # ★⑥ floodFill 把所有与室外连通的墙体整段抹掉
        # 先给 floodFill 提前准备 mask；floodFill 只能改“非零”种子
        mask2 = np.zeros((H + 2, W + 2), np.uint8)

        # floodFill 需要一个“种子像素”，但 candidate 里可能很多
        # 遍历 candidate==1 的所有像素，逐个 floodFill .
        ys, xs = np.where(candidate == 1)
        for y, x in zip(ys, xs):
            if wall_mask[y, x] == 1:  # 可能前面已被填 0，要判断
                cv2.floodFill(wall_mask, mask2, seedPoint=(int(x), int(y)), newVal=0)


    # 3️⃣ 第三阶段：像素级房间对标签打标
    # 创建与wall_array同形状的数组用于标记墙体像素对应的房间对ID
    room_pair_map = np.zeros_like(wall_array, dtype=np.int32)
    pair_index_map = {}  # 字典: (roomA, roomB) -> 唯一索引
    current_index = 1
    H, W = wall_array.shape
    # 遍历每个墙体像素
    for r in range(H):
        for c in range(W):
            if wall_array[r, c] == 0:
                continue  # 非墙体像素跳过
            # 收集此墙像素8邻域内相邻的房间ID（region_id_map > 0）
            neighbor_ids = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue  # 本身
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        rid = region_id_map[nr, nc]
                        if rid > 0:
                            neighbor_ids.add(int(rid))
            # 若相邻房间数不是恰好2（如墙端点或交叉点），则不标记房间对
            if len(neighbor_ids) != 2:
                continue
            # 恰有两个相邻房间
            roomA, roomB = sorted(list(neighbor_ids))
            pair = (roomA, roomB)
            # 为该房间对分配唯一索引
            if pair not in pair_index_map:
                pair_index_map[pair] = current_index
                current_index += 1
            pid = pair_index_map[pair]
            room_pair_map[r, c] = pid

    # 4️⃣ 第四阶段：基于“墙体连通性 + 房间对”聚类墙段
    # 先构建索引到房间对的映射，方便通过索引查回对应房间ID对
    index_pair_map = {idx: pair for pair, idx in pair_index_map.items()}
    H, W = room_pair_map.shape
    # 针对每一种房间对，提取其对应的墙体像素掩码，连通区域划分墙段
    for pid, pair in index_pair_map.items():
        # 提取当前房间对对应的墙像素掩码
        mask = (room_pair_map == pid).astype(np.uint8)
        if mask.sum() == 0:
            continue  # 无墙像素，跳过
        # 提取连通的墙段（使用8邻接确保对角连接的连续墙段也算作一段）
        num_labels, comp_label_img = cv2.connectedComponents(mask, connectivity=4)
        for lbl in range(1, num_labels):
            comp_coords = np.where(comp_label_img == lbl)
            if comp_coords[0].size == 0:
                continue
            comp_rows = comp_coords[0];
            comp_cols = comp_coords[1]
            # 计算墙段连通区域的边界框尺寸
            min_r = int(comp_rows.min());
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min());
            max_c = int(comp_cols.max())
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            length = max(height, width)
            thick = min(height, width)
            roomA, roomB = pair
            edges.append({
                'roomA': roomA, 'roomB': roomB,
                'type': 'wall',
                'length': int(length), 'width': int(thick)
            })

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
            comp_rows = comp_coords[0];
            comp_cols = comp_coords[1]
            # 计算开口段的边界框尺寸
            min_r = int(comp_rows.min());
            max_r = int(comp_rows.max())
            min_c = int(comp_cols.min());
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
