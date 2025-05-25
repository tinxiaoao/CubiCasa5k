import numpy as np
from scipy.ndimage import label, binary_dilation


def detect_adjacency(region_id_map, wall_array, door_array=None, window_array=None):
    """
    Detect adjacency between regions based on wall and door/window masks.
    Returns a dictionary `edges` where each key is a tuple of two region IDs and
    the value is a dict with fields:
      - connection_types: set of connection types ('wall', 'door', 'window') between the two regions
      - num_door_window: number of door/window connections between the regions
      - area_door_window: total pixel area of door/window connections
      - num_wall: number of wall segments between the regions
      - area_wall: total pixel area of wall segments
    """
    edges = {}

    # 1. Identify connected wall segments using scipy.ndimage.label
    wall_labels, num_walls = label(wall_array)
    if num_walls > 0:
        # Pre-compute pixel area of each wall segment
        wall_counts = np.bincount(wall_labels.ravel())
        # Define 4-connectivity structure for neighbor detection (up, down, left, right)
        structure = np.array([[False, True, False],
                              [True, True, True],
                              [False, True, False]], dtype=bool)
        # Process each wall segment
        for wall_id in range(1, num_walls + 1):
            # Extract the mask for this wall segment
            segment_mask = (wall_labels == wall_id)
            if not segment_mask.any():
                continue
            # Find all neighboring pixels (4-neighborhood) around this wall segment
            dilated_mask = binary_dilation(segment_mask, structure=structure)
            neighbor_mask = np.logical_and(dilated_mask, ~segment_mask)
            neighbor_vals = region_id_map[neighbor_mask]
            if neighbor_vals.size == 0:
                continue
            neighbors = set(np.unique(neighbor_vals))
            # 2. Check if the wall segment neighbors two or more different rooms
            # (regions with id > 0, excluding background)
            skip_segment = False
            # 2.a. If any boundary neighbor is background or outdoor (region_id 0 or 1), skip this wall segment
            if 0 in neighbors or 1 in neighbors:
                skip_segment = True
            # Remove background/outdoor from neighbor set for room count
            neighbors.discard(0)
            neighbors.discard(1)
            if skip_segment:
                # Do not include this wall segment in room connections
                continue
            # If the wall segment is adjacent to two or more different non-background regions
            if len(neighbors) >= 2:
                # Calculate area (pixel count) of this wall segment
                segment_area = int(wall_counts[wall_id] if wall_id < len(wall_counts) else np.sum(segment_mask))
                neighbors_list = sorted(neighbors)
                # 2.b. Record this wall segment as a connection (edge) between each pair of adjacent regions
                for i in range(len(neighbors_list)):
                    for j in range(i + 1, len(neighbors_list)):
                        r1, r2 = neighbors_list[i], neighbors_list[j]
                        if r1 == r2:
                            continue
                        edge_key = (r1, r2) if r1 < r2 else (r2, r1)
                        if edge_key not in edges:
                            edges[edge_key] = {
                                'connection_types': set(),
                                'num_door_window': 0,
                                'area_door_window': 0,
                                'num_wall': 0,
                                'area_wall': 0
                            }
                        edges[edge_key]['connection_types'].add('wall')
                        edges[edge_key]['num_wall'] += 1
                        edges[edge_key]['area_wall'] += segment_area

    # 3. Door/window connection logic (unchanged)
    if door_array is not None or window_array is not None:
        # Combine door and window masks if both are provided
        if door_array is not None and window_array is not None:
            door_window_mask = np.logical_or(door_array, window_array)
        elif door_array is not None:
            door_window_mask = door_array.copy()
        else:
            door_window_mask = window_array.copy()
        # Label connected components of door/window openings
        opening_labels, num_openings = label(door_window_mask)
        if num_openings > 0:
            opening_counts = np.bincount(opening_labels.ravel())
            structure = np.array([[False, True, False],
                                  [True, True, True],
                                  [False, True, False]], dtype=bool)
            # Process each door/window opening segment
            for open_id in range(1, num_openings + 1):
                segment_mask = (opening_labels == open_id)
                if not segment_mask.any():
                    continue
                dilated_mask = binary_dilation(segment_mask, structure=structure)
                neighbor_mask = np.logical_and(dilated_mask, ~segment_mask)
                neighbor_vals = region_id_map[neighbor_mask]
                if neighbor_vals.size == 0:
                    continue
                neighbors = set(np.unique(neighbor_vals))
                # Remove background (0) from neighbors for connectivity check
                neighbors.discard(0)
                # (We do NOT skip segments touching outdoor (1) for door/window, keeping exterior door connections)
                if len(neighbors) >= 2:
                    # Calculate area of this opening segment
                    segment_area = int(
                        opening_counts[open_id] if open_id < len(opening_counts) else np.sum(segment_mask))
                    # Determine whether this segment is a door or window (or combined)
                    conn_type = None
                    if door_array is not None and np.any(np.logical_and(segment_mask, door_array)):
                        conn_type = 'door'
                    if window_array is not None and np.any(np.logical_and(segment_mask, window_array)):
                        if conn_type is None:
                            conn_type = 'window'
                        else:
                            # If a segment contains both door and window pixels (unlikely), treat it as 'door'
                            conn_type = 'door'
                    if conn_type is None:
                        conn_type = 'door_window'
                    neighbors_list = sorted(neighbors)
                    # Record this opening as a connection between each pair of adjacent regions
                    for i in range(len(neighbors_list)):
                        for j in range(i + 1, len(neighbors_list)):
                            r1, r2 = neighbors_list[i], neighbors_list[j]
                            if r1 == r2:
                                continue
                            edge_key = (r1, r2) if r1 < r2 else (r2, r1)
                            if edge_key not in edges:
                                edges[edge_key] = {
                                    'connection_types': set(),
                                    'num_door_window': 0,
                                    'area_door_window': 0,
                                    'num_wall': 0,
                                    'area_wall': 0
                                }
                            edges[edge_key]['connection_types'].add(conn_type)
                            edges[edge_key]['num_door_window'] += 1
                            edges[edge_key]['area_door_window'] += segment_area

    return edges
