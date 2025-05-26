import numpy as np
from scipy.ndimage import binary_erosion, label, binary_dilation


def get_boundary_mask(region_id_map, rid):
    """
    返回房间 rid 的边界像素布尔掩膜。
    边界像素定义：属于房间 rid 且至少有一个 4‑邻接像素不属于 rid 的像素。
    """
    room_mask = (region_id_map == rid)
    if not room_mask.any():
        return np.zeros_like(region_id_map, dtype=bool)

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    interior = binary_erosion(room_mask, structure=structure, border_value=0)
    return room_mask & ~interior


def detect_adjacency(region_id_map, wall_array, icon_array, wall_label_array, debug=False):
    """核心逻辑完全继承自 v0，仅在**墙段最终写入 edges 之前**增加
    『同一段墙最多生成一条连接，若对应多对房间则保留面积最小者』  的筛选。
    其它行为（门/窗检测、外墙排除、逐像素扫描）保持不变。
    """
    edges = {}

    def ensure_edge(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in edges:
            edges[k] = {
                "connection_types": set(),
                "num_door_window": 0,
                "area_door_window": 0,
                "num_wall": 0,
                "area_wall": 0,
            }
        return k

    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

    # ---------- 1. 门 / 窗 连接（原样保留） ----------
    for val, tp in ((1, "door"), (2, "window")):
        mask = icon_array == val
        lab, n = label(mask, structure=struct)
        for i in range(1, n + 1):
            comp = lab == i
            if not comp.any():
                continue
            dil = binary_dilation(comp, structure=struct, border_value=0)
            neigh = np.unique(region_id_map[dil & ~comp])
            neigh = neigh[neigh > 0]
            if neigh.size == 2:
                a, b = int(neigh[0]), int(neigh[1])
                k = ensure_edge(a, b)
                edges[k]["connection_types"].add(tp)
                edges[k]["num_door_window"] += 1
                edges[k]["area_door_window"] += int(comp.sum())

    # ---------- 2. 墙体连接 ----------
    exclude_idx = {0, 1, 8, 50}
    visited = np.zeros_like(wall_array, bool)
    h, w = wall_array.shape

    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]

    for rid in region_ids:
        bound = get_boundary_mask(region_id_map, rid)
        bx, by = np.nonzero(bound)
        for x, y in zip(bx, by):
            for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < h and 0 <= ny < w):
                    continue
                if wall_array[nx, ny] != 1 or visited[nx, ny]:
                    continue

                seg = []  # 当前墙段像素
                t = 0
                other = None  # 另一房间id
                last = None
                while True:
                    cx, cy = x + dx * t, y + dy * t
                    if not (0 <= cx < h and 0 <= cy < w):
                        break
                    if wall_array[cx, cy] == 1:
                        seg.append((cx, cy))
                        t += 1
                        continue
                    if region_id_map[cx, cy] not in (0, rid):
                        other = int(region_id_map[cx, cy])
                        last = (cx - dx, cy - dy)
                    break

                if other is None:
                    continue

                # 延伸两端（同 v0）
                seg_set = set(seg)
                visited[list(zip(*seg))] = True
                base_x, base_y = last if last else seg[-1]
                if dx == 0:
                    orient = ((1, 0), (-1, 0))
                    sideA, sideB = ((0, -1), (0, 1)) if dy == 1 else ((0, 1), (0, -1))
                else:
                    orient = ((0, 1), (0, -1))
                    sideA, sideB = ((-1, 0), (1, 0)) if dx == 1 else ((1, 0), (-1, 0))

                for odx, ody in orient:
                    cx, cy = base_x, base_y
                    while True:
                        cx += odx
                        cy += ody
                        if not (0 <= cx < h and 0 <= cy < w):
                            break
                        if wall_array[cx, cy] != 1 or visited[cx, cy]:
                            break
                        ax, ay = cx + sideA[0], cy + sideA[1]
                        bx, by = cx + sideB[0], cy + sideB[1]
                        if not (0 <= ax < h and 0 <= ay < w and 0 <= bx < h and 0 <= by < w):
                            break
                        if region_id_map[ax, ay] != rid or region_id_map[bx, by] != other:
                            break
                        seg_set.add((cx, cy))
                        visited[cx, cy] = True

                # 外墙排除
                ext = False
                for sx, sy in seg_set:
                    for adx, ady in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        qx, qy = sx + adx, sy + ady
                        if 0 <= qx < h and 0 <= qy < w and wall_label_array[qx, qy] in exclude_idx:
                            ext = True;
                            break
                    if ext:
                        break
                if ext:
                    continue

                # ============ 新增：同段墙只留最小面积 pair ============
                pair_count = {}
                for sx, sy in seg_set:
                    # 左右
                    if sy - 1 >= 0 and sy + 1 < w:
                        a = region_id_map[sx, sy - 1]
                        b = region_id_map[sx, sy + 1]
                        if a > 0 and b > 0 and a != b:
                            p = (a, b) if a < b else (b, a)
                            pair_count[p] = pair_count.get(p, 0) + 1
                    # 上下
                    if sx - 1 >= 0 and sx + 1 < h:
                        a = region_id_map[sx - 1, sy]
                        b = region_id_map[sx + 1, sy]
                        if a > 0 and b > 0 and a != b:
                            p = (a, b) if a < b else (b, a)
                            pair_count[p] = pair_count.get(p, 0) + 1
                if not pair_count:
                    continue
                best_pair, best_area = min(pair_count.items(), key=lambda kv: kv[1])

                k = ensure_edge(*best_pair)
                edges[k]["connection_types"].add("wall")
                edges[k]["num_wall"] += 1
                edges[k]["area_wall"] += best_area
    return edges
