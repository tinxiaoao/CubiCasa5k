import numpy as np
from collections import defaultdict
from scipy.ndimage import label, binary_dilation


# -----------------------------------------------------------------------------
# 该版本只重写“墙体连接”检测部分；门 / 窗连接部分保持 detect_adjacency_v0.py 原样。
# -----------------------------------------------------------------------------

def get_boundary_mask(region_id_map: np.ndarray, rid: int) -> np.ndarray:
    """返回房间 rid 的边界像素布尔掩码。"""

    room_mask = (region_id_map == rid)
    if room_mask.sum() == 0:
        return np.zeros_like(room_mask, dtype=bool)

    # 4-邻接结构用卷积近似 erosion 得到 interior
    # faster than binary_erosion for simple 4‑neighbours mask
    interior = (
            np.roll(room_mask, 1, axis=0) & np.roll(room_mask, -1, axis=0) &
            np.roll(room_mask, 1, axis=1) & np.roll(room_mask, -1, axis=1) & room_mask
    )
    return room_mask & (~interior)


# =============================================================================
# 主函数 detect_adjacency
# =============================================================================

def detect_adjacency(region_id_map: np.ndarray,
                     wall_array: np.ndarray,
                     icon_array: np.ndarray,
                     *,
                     wall_label_raw: np.ndarray = None,
                     exclude_indices: set = {0, 1, 8, 50}) -> dict:
    """检测房间之间的门/窗 + 墙体连接。

    Parameters
    ----------
    region_id_map : np.ndarray
        每个像素的房间 id（0 表示非房间）。
    wall_array : np.ndarray
        墙体二值图：墙体像素==1；其他==0。
    icon_array : np.ndarray
        图标图：门==1，窗==2，其余==0。
    wall_label_raw : np.ndarray, optional
        原始索引图（一般可直接用 np.array(Image.open('wall_svg.png')) 读取得到）。
        用于判断墙段是否接触 exclude_indices。若为空，则回退到 region_id_map==0 的检测方式。
    exclude_indices : set, default {0,1,8,50}
        与这些索引相邻的墙段视为外墙并被忽略。

    Returns
    -------
    edges : dict
        {(id1,id2): {connection_types,set,...}}
    """
    H, W = region_id_map.shape
    edges = defaultdict(lambda: {
        'connection_types': set(),
        'num_door_window': 0,
        'area_door_window': 0,
        'num_wall': 0,
        'area_wall': 0,
    })

    # ------------------------------------------------------------------
    # 1) 门 / 窗连接（保持 v0 逻辑）
    # ------------------------------------------------------------------
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for icon_val, icon_type in ((1, 'door'), (2, 'window')):
        icon_mask = (icon_array == icon_val)
        labeled_icons, num_icons = label(icon_mask, structure=structure)
        for lbl in range(1, num_icons + 1):
            comp_mask = (labeled_icons == lbl)
            if not comp_mask.any():
                continue
            # 门窗膨胀 1 像素得到邻接区域
            neighbour = binary_dilation(comp_mask, structure=structure) & (~comp_mask)
            neigh_ids = np.unique(region_id_map[neighbour])
            neigh_ids = neigh_ids[neigh_ids > 0]
            if neigh_ids.size == 2:
                id1, id2 = sorted(map(int, neigh_ids))
                e = edges[(id1, id2)]
                e['connection_types'].add(icon_type)
                e['num_door_window'] += 1
                e['area_door_window'] += int(comp_mask.sum())

    # ------------------------------------------------------------------
    # 2) 墙体连接：连通块 + 外墙排除

    # ------------------------------------------------------------------
    # 2‑1 连通块标记
    wall_labels, num_wall_cc = label(wall_array, structure=structure)

    # 预先准备每个墙段接触到的房间集合 & 排除标记
    cc_touch_rooms = [set() for _ in range(num_wall_cc + 1)]  # 0 unused
    cc_touch_exclude = [False] * (num_wall_cc + 1)

    # 遍历墙体像素一次性统计相邻信息（比逐 CC 遍历快）
    # 通过查看四邻接 room_id & raw_label
    up_rid = np.roll(region_id_map, 1, axis=0)
    down_rid = np.roll(region_id_map, -1, axis=0)
    left_rid = np.roll(region_id_map, 1, axis=1)
    right_rid = np.roll(region_id_map, -1, axis=1)

    if wall_label_raw is None:
        # 若未提供 raw label，则用 region_id_map==0 作为 exclude 判断
        up_lab = np.zeros_like(region_id_map)
        down_lab = np.zeros_like(region_id_map)
        left_lab = np.zeros_like(region_id_map)
        right_lab = np.zeros_like(region_id_map)
    else:
        up_lab = np.roll(wall_label_raw, 1, axis=0)
        down_lab = np.roll(wall_label_raw, -1, axis=0)
        left_lab = np.roll(wall_label_raw, 1, axis=1)
        right_lab = np.roll(wall_label_raw, -1, axis=1)

    # 当前墙像素坐标
    wy, wx = np.nonzero(wall_array)
    for y, x in zip(wy, wx):
        lbl = wall_labels[y, x]
        # 相邻房间 id
        for rid in (region_id_map[y, x], up_rid[y, x], down_rid[y, x], left_rid[y, x], right_rid[y, x]):
            if rid > 0:
                cc_touch_rooms[lbl].add(int(rid))
        # 相邻排除类别索引
        for lab in (up_lab[y, x], down_lab[y, x], left_lab[y, x], right_lab[y, x]):
            if lab in exclude_indices:
                cc_touch_exclude[lbl] = True

    # 2‑2 遍历 CC 判定内部隔墙
    for lbl in range(1, num_wall_cc + 1):
        rooms_touched = cc_touch_rooms[lbl]
        if len(rooms_touched) < 2 or cc_touch_exclude[lbl]:
            # 外墙或仅触一个房间 → skip
            continue
        # 有效内部隔墙：对房间对做两两组合
        rooms_list = sorted(rooms_touched)
        area = int((wall_labels == lbl).sum())
        for i in range(len(rooms_list)):
            for j in range(i + 1, len(rooms_list)):
                id1, id2 = rooms_list[i], rooms_list[j]
                e = edges[(id1, id2)]
                if 'wall' not in e['connection_types']:
                    e['connection_types'].add('wall')
                    e['num_wall'] += 1  # 该墙段只计一次
                    e['area_wall'] += area
                # 如果之前门窗连接已存在，只是附加 'wall' 类型即可

    return dict(edges)
