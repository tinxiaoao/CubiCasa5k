import numpy as np


def extract_rooms(wall_array: np.ndarray):
    """
    从墙体图像数组中提取房间区域。
    返回值：
    - region_id_map: 与wall_array大小相同的数组，每个房间像素填入房间ID，非房间区域为0。
    - rooms: 列表，每个元素是一个房间的属性字典，包括id, area, room_type。
    """
    # 排除的类别索引
    exclude_indices = {0, 1, 2, 8, 50}
    h, w = wall_array.shape
    # 构建二值掩膜：房间内部像素=1，排除区域=0
    mask = np.ones((h, w), dtype=np.uint8)
    mask[np.isin(wall_array, list(exclude_indices))] = 0

    # 连通域标记（4连通）
    region_id_map = np.zeros_like(mask, dtype=np.int32)
    current_id = 0
    # 手动实现简单的DFS/BFS连通域标记，也可以用 cv2.connectedComponents
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 1 and region_id_map[i, j] == 0:
                current_id += 1
                # Flood-fill BFS/DFS
                stack = [(i, j)]
                region_id_map[i, j] = current_id
                # 用栈进行DFS填充
                while stack:
                    x, y = stack.pop()
                    # 检查邻居像素（4连通）
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < h and 0 <= ny_ < w:
                            if mask[nx_, ny_] == 1 and region_id_map[nx_, ny_] == 0:
                                region_id_map[nx_, ny_] = current_id
                                stack.append((nx_, ny_))
    num_rooms = current_id

    # 准备类别映射（主要房间类型）
    category_to_room = {
        3: "Kitchen",  # 厨房
        4: "Living Room",  # 客厅
        5: "Bedroom",  # 卧室
        6: "Bath",  # 卫生间/浴室
        7: "Hallway",  # 门厅/走廊
        9: "Storage",  # 储藏室/衣帽间等
        10: "Garage",  # 车库
    }
    rooms = []
    # 逐个房间计算面积和类型
    for rid in range(1, num_rooms + 1):
        # 提取该房间区域像素的原始类别索引
        region_mask = (region_id_map == rid)
        area = int(region_mask.sum())
        # 统计区域内各类别频率
        vals, counts = np.unique(wall_array[region_mask], return_counts=True)
        # 排除掉区域内可能残留的非房间类别（例如房间内家具类别，但这些也算房间内容）
        # 在统计中，我们直接选择出现最多的类别作为房间类别
        if len(vals) > 0:
            # 去掉排除类别再判断最大频率
            # （如果房间内存在少量墙体像素或透明像素边缘，可能也被包括在区域边缘，可以忽略）
            filtered = [(val, cnt) for val, cnt in zip(vals, counts) if val not in exclude_indices]
            if not filtered:
                # 若filtered为空，默认类别未知
                main_category = None
            else:
                # 按像素数排序，选择最大者
                main_category = max(filtered, key=lambda x: x[1])[0]
        else:
            main_category = None

        # 确定房间类型名称
        if main_category is None:
            room_type = "Other"
        elif main_category in category_to_room:
            room_type = category_to_room[main_category]
        else:
            # 其他未列出的类别都视作Other（可能是家具等，意味着缺少明确房间地面类别）
            room_type = "Other"
        rooms.append({"id": rid, "area": area, "room_type": room_type})
    return region_id_map, rooms
