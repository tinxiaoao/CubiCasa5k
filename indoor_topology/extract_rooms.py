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
    exclude_indices = {0, 1, 2, 8, 50}  # 室外/墙/背景/透明
    unique_room_indices = np.unique(wall_array)
    room_indices = [idx for idx in unique_room_indices if idx not in exclude_indices]

    region_id_map = np.zeros_like(wall_array, dtype=np.int32)
    current_label = 1

    rooms = []
    min_avg_room_length = float("inf")
    wall_outside_width = float("inf")
    kernel = np.ones((3, 3), np.uint8)

    category_to_room = {
        3: "Kitchen", 4: "Living Room", 5: "Bedroom",
        6: "Bath", 7: "Hallway", 9: "Storage", 10: "Garage",
    }

    # 修改连通域分析逻辑
    for room_idx in room_indices:
        mask = (wall_array == room_idx).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

        for lbl in range(1, num_labels):
            region_mask = (labels == lbl)
            area = int(region_mask.sum())

            vals, counts = np.unique(wall_array[region_mask], return_counts=True)
            filtered = [(v, c) for v, c in zip(vals, counts) if v not in exclude_indices]
            main_category = max(filtered, key=lambda x: x[1])[0] if filtered else None
            room_type = category_to_room.get(main_category, "Other")

            boundary = cv2.morphologyEx(region_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
            perimeter = int(boundary.sum())
            avg_len = perimeter / 10
            min_avg_room_length = min(min_avg_room_length, avg_len)
            wall_outside_width = perimeter / 10

            rooms.append({
                "id": current_label,
                "area": area,
                "perimeter": perimeter,
                "room_type": room_type
            })

            region_id_map[region_mask] = current_label
            current_label += 1

    if min_avg_room_length == float("inf"):
        min_avg_room_length = 0.0
        wall_outside_width = 0.0

    return region_id_map, rooms, float(min_avg_room_length), float(wall_outside_width)
