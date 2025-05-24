def detect_adjacency(region_id_map, wall_array, icon_array):
    """
    Detects adjacency between rooms via doors, windows, and walls.
    Returns a dict where keys are (room1, room2) tuples and values contain:
      - 'connection_types': set of {'door', 'window', 'wall'} indicating types of connections.
      - 'num_door_window': count of door/window openings between the rooms.
      - 'area_door_window': total area (in pixels) of those door/window openings.
      - 'num_wall': count of wall segments between the rooms.
      - 'area_wall': total area (pixels) of those wall segments.
    """
    import numpy as np
    from collections import deque
    from scipy.ndimage import label, binary_dilation, binary_erosion

    edges = {}

    def ensure_edge_entry(a, b):
        """Ensure the edge dictionary has an entry for the room pair (a, b)."""
        key = (a, b) if a < b else (b, a)
        if key not in edges:
            edges[key] = {
                'connection_types': set(),
                'num_door_window': 0,
                'area_door_window': 0,
                'num_wall': 0,
                'area_wall': 0
            }
        return key

    # 4-connected structuring element (cross-shaped) for labeling and dilation
    structure = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=bool)

    # **Door and Window connections**
    for icon_val, icon_type in [(1, 'door'), (2, 'window')]:
        icon_mask = (icon_array == icon_val)
        labeled_icons, num_icons = label(icon_mask, structure=structure)
        for lbl in range(1, num_icons + 1):
            comp_mask = (labeled_icons == lbl)
            if not comp_mask.any():
                continue
            # Dilate the icon region to find adjacent rooms
            dilated = binary_dilation(comp_mask, structure=structure, border_value=0)
            neighbor_area = dilated & ~comp_mask  # area just around the icon
            neighbor_ids = np.unique(region_id_map[neighbor_area])
            neighbor_ids = neighbor_ids[neighbor_ids > 0]  # exclude background (0)
            neighbor_ids = np.unique(neighbor_ids)
            if neighbor_ids.size == 2:
                a, b = int(neighbor_ids[0]), int(neighbor_ids[1])
                key = ensure_edge_entry(a, b)
                edges[key]['connection_types'].add(icon_type)
                edges[key]['num_door_window'] += 1
                # Use the mask size (pixels count) as the opening area
                edges[key]['area_door_window'] += int(comp_mask.sum())

    # **Wall connections**
    # Get all room IDs (exclude 0 for background)
    region_ids = np.unique(region_id_map)
    region_ids = region_ids[region_ids != 0]

    # Visited mask for wall pixels to avoid double counting
    visited_wall = np.zeros_like(wall_array, dtype=bool)

    # Helper to get boundary pixels mask for a given room id
    def get_boundary_mask(rid):
        """Returns a boolean mask of boundary pixels for room `rid`."""
        room_mask = (region_id_map == rid)
        if not room_mask.any():
            return np.zeros_like(region_id_map, dtype=bool)
        interior = binary_erosion(room_mask, structure=structure, border_value=0)
        boundary_mask = room_mask & ~interior
        return boundary_mask

    # Check each room's boundary for wall connectivity
    for rid in region_ids:
        boundary_mask = get_boundary_mask(rid)
        boundary_coords = np.transpose(np.nonzero(boundary_mask))
        # Iterate over each boundary pixel of room rid
        for x, y in boundary_coords:
            # Check four orthogonal neighbors (up, down, left, right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                # Skip if neighbor is out of bounds or not a wall pixel
                if nx < 0 or nx >= region_id_map.shape[0] or ny < 0 or ny >= region_id_map.shape[1]:
                    continue
                if wall_array[nx, ny] != 1:
                    continue
                # If this wall pixel was already accounted for, skip it
                if visited_wall[nx, ny]:
                    continue

                # Neighbor is a wall pixel and not visited: start BFS path search
                # Only consider paths fully within walls connecting room rid to another room
                # Analyze the starting wall pixel's neighbors to enforce rules
                neighbor_regions = set()
                is_outer = False
                for adx, ady in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    ax, ay = nx + adx, ny + ady
                    if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                        # Touches image boundary, considered outside
                        is_outer = True
                        break
                    if wall_array[ax, ay] == 1:
                        # Adjacent wall pixel, ignore for region check
                        continue
                    rid_neighbor = region_id_map[ax, ay]
                    if rid_neighbor == 0:
                        # Neighbor is background (outside wall structure)
                        is_outer = True
                        break
                    if rid_neighbor != rid:
                        neighbor_regions.add(rid_neighbor)
                # Skip this wall pixel if it contacts outside or more than one different room (intersection)
                if is_outer or len(neighbor_regions) > 1:
                    visited_wall[nx, ny] = True
                    continue

                # Initialize BFS for this wall segment
                encountered_room = None
                allowed_rooms = {rid}
                # If exactly one other room neighbor found, set it as the target room
                if len(neighbor_regions) == 1:
                    encountered_room = neighbor_regions.pop()
                    allowed_rooms.add(encountered_room)

                dq = deque()
                dq.append((nx, ny))
                visited_wall[nx, ny] = True
                wall_segment_pixels = []  # record pixels in this wall segment path

                # BFS through wall pixels
                while dq:
                    cx, cy = dq.popleft()
                    wall_segment_pixels.append((cx, cy))
                    # Examine neighboring wall pixels
                    for odx, ody in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        wx, wy = cx + odx, cy + ody
                        # Skip if out of bounds or not a wall
                        if wx < 0 or wx >= region_id_map.shape[0] or wy < 0 or wy >= region_id_map.shape[1]:
                            continue
                        if wall_array[wx, wy] != 1 or visited_wall[wx, wy]:
                            continue
                        # Check this wall pixel's adjacent regions for validity
                        nbr_regions = set()
                        outer_flag = False
                        for adx, ady in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            ax, ay = wx + adx, wy + ady
                            if ax < 0 or ax >= region_id_map.shape[0] or ay < 0 or ay >= region_id_map.shape[1]:
                                outer_flag = True
                                break
                            if wall_array[ax, ay] == 1:
                                continue  # ignore adjacent wall pixels
                            rid_nbr = region_id_map[ax, ay]
                            if rid_nbr == 0:
                                outer_flag = True
                                break
                            if rid_nbr != rid:
                                nbr_regions.add(rid_nbr)
                        # If touching outside, or invalid neighbor regions, skip this wall pixel
                        if outer_flag:
                            continue
                        # Remove current room id from neighbors (we only care about other rooms)
                        nbr_regions.discard(rid)
                        if encountered_room is None:
                            # No target encountered yet
                            if len(nbr_regions) == 0:
                                # Still within walls adjacent only to room rid
                                pass
                            elif len(nbr_regions) == 1:
                                # Found exactly one new room neighbor -> set as target
                                new_room = next(iter(nbr_regions))
                                encountered_room = new_room
                                allowed_rooms.add(new_room)
                            else:
                                # More than one new room encountered (intersection) -> stop this path
                                continue
                        else:
                            # Already have a target room
                            if len(nbr_regions) == 0:
                                # Adjacent only to rid (and wall) - continue
                                pass
                            elif len(nbr_regions) == 1:
                                # Adjacent to one room
                                other_room = next(iter(nbr_regions))
                                if other_room != encountered_room:
                                    # A different room appears - invalid path
                                    continue
                            else:
                                # Adjacent to more than one room (invalid)
                                continue
                        # Enqueue this wall pixel as part of the current wall segment path
                        dq.append((wx, wy))
                        visited_wall[wx, wy] = True

                # BFS complete for this segment
                if encountered_room is None:
                    # No second room found, not a valid adjacency between two rooms
                    continue

                # We found a wall connection between rid and encountered_room
                key = ensure_edge_entry(rid, encountered_room)
                edges[key]['connection_types'].add('wall')
                edges[key]['num_wall'] += 1
                # Calculate wall segment area (number of wall pixels in this path segment)
                segment_pixel_count = len(set(wall_segment_pixels))
                edges[key]['area_wall'] += segment_pixel_count

    return edges
