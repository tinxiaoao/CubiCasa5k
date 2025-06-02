import numpy as np
import cv2


def extract_rooms(wall_array: np.ndarray):
    """
    输入:
        wall_array : 调色板索引图 (H, W)。房间像素!=exclude_indices，其余视为室外/墙等
    输出:
        region_id_map            : (H, W) 房间ID, 非房间0
        rooms                    : List[Dict] [{'id', 'area', 'room_type'}]
        min_avg_room_length (float) : 所有房间中最小的“平均边长”(perimeter/4)
    """
    # ------------------------------------------------------------------
    # 1️⃣ 生成房间掩膜
    # ------------------------------------------------------------------
    exclude_indices = {0, 1, 2, 8, 50}  # 室外/墙/背景/透明
    mask = (~np.isin(wall_array, list(exclude_indices))).astype(np.uint8)  # 房间=1

    # ------------------------------------------------------------------
    # 2️⃣ 连通域标记 (8-连通) → 得到 region_id_map
    # ------------------------------------------------------------------
    num_labels, region_id_map = cv2.connectedComponents(mask, connectivity=8)
    # num_labels 含背景0; 房间ID范围 1..num_labels-1

    # ------------------------------------------------------------------
    # 3️⃣ 遍历房间, 计算 area, room_type, perimeter, 更新 min_avg_len
    # ------------------------------------------------------------------
    category_to_room = {
        3: "Kitchen", 4: "Living Room", 5: "Bedroom",
        6: "Bath", 7: "Hallway", 9: "Storage", 10: "Garage",
    }
    rooms = []
    min_avg_room_length = float("inf")
    kernel = np.ones((3, 3), np.uint8)

    for rid in range(1, num_labels):
        region_mask = (region_id_map == rid)
        if not region_mask.any():
            continue
        area = int(region_mask.sum())

        # --- 房间类别判定（与旧版保持一致） ---
        vals, counts = np.unique(wall_array[region_mask], return_counts=True)
        filtered = [(v, c) for v, c in zip(vals, counts) if v not in exclude_indices]
        main_category = max(filtered, key=lambda x: x[1])[0] if filtered else None
        room_type = category_to_room.get(main_category, "Other")

        # --- 周长 -> 平均边长 ---
        boundary = cv2.morphologyEx(region_mask.astype(np.uint8),
                                    cv2.MORPH_GRADIENT, kernel)
        perimeter = int(boundary.sum())  # 像素级周长
        avg_len = perimeter / 10  # 平均边长估算 设为4的话，在房间较大的平面图中容易出现误判，
        min_avg_room_length = min(min_avg_room_length, avg_len)

        rooms.append({
            "id": rid,
            "area": area,
            "room_type": room_type
        })

    # 若无房间, 置 0
    if min_avg_room_length == float("inf"):
        min_avg_room_length = 0.0

    return region_id_map.astype(np.int32), rooms, float(min_avg_room_length)
