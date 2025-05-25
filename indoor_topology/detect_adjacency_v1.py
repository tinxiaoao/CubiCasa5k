import numpy as np
from scipy.ndimage import binary_dilation, label

def detect_adjacency(region_id_map, wall_array, icon_array):
    """
    识别平面图中房间之间通过门、窗或墙的连接关系。

    参数:
        region_id_map (ndarray): 2D数组，每个像素值代表房间ID（背景为0）
        wall_array (ndarray): 2D数组，墙体像素为1，非墙为0
        icon_array (ndarray): 2D数组，门窗图标标记（1=门，2=窗，0=无）

    返回:
        edges字典:
            键为 (房间ID1, 房间ID2)
            值为字典:
              - 'connection_types': set(['door', 'window', 'wall'])
              - 'num_door_window': 门窗连接段数
              - 'area_door_window': 门窗连接的总像素面积
              - 'num_wall': 墙连接段数
              - 'area_wall': 墙连接的总像素面积
    """
    edges = {}

    def init_edge_entry(id1, id2):
        """确保房间对在edges字典中已初始化。"""
        pair = tuple(sorted((int(id1), int(id2))))
        if pair not in edges:
            edges[pair] = {
                'connection_types': set(),
                'num_door_window': 0,
                'area_door_window': 0,
                'num_wall': 0,
                'area_wall': 0
            }
        return pair

    # 🌟 门窗连接检测逻辑
    icon_mask = (icon_array > 0)
    icon_mask_dilated = binary_dilation(icon_mask, structure=np.ones((3, 3), bool))
    labeled_icons, num_icon_clusters = label(icon_mask_dilated, structure=np.ones((3, 3), bool))

    for label_id in range(1, num_icon_clusters + 1):
        cluster_mask = (labeled_icons == label_id)
        if not cluster_mask.any():
            continue

        region_neighbors = set()
        cluster_coords = np.argwhere(cluster_mask)

        for (x, y) in cluster_coords:
            # 检查四邻域像素，记录房间ID
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < region_id_map.shape[0] and 0 <= ny < region_id_map.shape[1]:
                    rid = region_id_map[nx, ny]
                    if rid > 0:
                        region_neighbors.add(int(rid))

        if len(region_neighbors) >= 2:
            neighbor_list = sorted(region_neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i+1, len(neighbor_list)):
                    id1, id2 = neighbor_list[i], neighbor_list[j]
                    pair = init_edge_entry(id1, id2)
                    cluster_icon_vals = icon_array[cluster_mask]
                    if 1 in cluster_icon_vals:
                        edges[pair]['connection_types'].add('door')
                    if 2 in cluster_icon_vals:
                        edges[pair]['connection_types'].add('window')
                    edges[pair]['num_door_window'] += 1
                    edges[pair]['area_door_window'] += int(cluster_mask.sum())

    # 🌟 墙体连接检测逻辑（重点升级部分）
    labeled_walls, num_wall_clusters = label(wall_array.astype(bool), structure=np.array([[0,1,0],
                                                                                         [1,1,1],
                                                                                         [0,1,0]], bool))
    H, W = region_id_map.shape
    for cid in range(1, num_wall_clusters + 1):
        cluster_mask = (labeled_walls == cid)
        if not cluster_mask.any():
            continue
        cluster_coords = np.argwhere(cluster_mask)

        # 排除与图像边界相连的墙（外墙）
        if np.any(cluster_coords[:,0] == 0) or np.any(cluster_coords[:,0] == H-1) or \
           np.any(cluster_coords[:,1] == 0) or np.any(cluster_coords[:,1] == W-1):
            continue

        neighbor_ids = set()
        for (x, y) in cluster_coords:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    rid = region_id_map[nx, ny]
                    if rid > 0:
                        neighbor_ids.add(int(rid))

        if len(neighbor_ids) < 2:
            continue

        neighbor_list = sorted(neighbor_ids)
        cluster_area = int(cluster_mask.sum())

        for i in range(len(neighbor_list)):
            for j in range(i+1, len(neighbor_list)):
                id1, id2 = neighbor_list[i], neighbor_list[j]
                # 检测两个房间是否被墙直接隔开
                direct_contact = False
                for (x, y) in cluster_coords:
                    if (0 < y < W-1 and
                        ((region_id_map[x, y-1]==id1 and region_id_map[x, y+1]==id2) or
                         (region_id_map[x, y-1]==id2 and region_id_map[x, y+1]==id1))):
                        direct_contact = True
                        break
                    if (0 < x < H-1 and
                        ((region_id_map[x-1, y]==id1 and region_id_map[x+1, y]==id2) or
                         (region_id_map[x-1, y]==id2 and region_id_map[x+1, y]==id1))):
                        direct_contact = True
                        break
                # 当只有两个房间邻接且没有严格对称像素对，也视作直接隔墙
                if not direct_contact and len(neighbor_ids) == 2:
                    direct_contact = True
                if direct_contact:
                    pair = init_edge_entry(id1, id2)
                    edges[pair]['connection_types'].add('wall')
                    edges[pair]['num_wall'] += 1
                    edges[pair]['area_wall'] += cluster_area

    return edges
