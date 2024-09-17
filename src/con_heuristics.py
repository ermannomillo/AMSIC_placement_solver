import device_util as deu
import IO_util as iou
from cffi import FFI
import numpy as np

ffi = FFI()

# Define the C function signature
ffi.cdef("""
    typedef struct {
        int x, y, w, h;
        int rect_idx;
        int pvar;
    } Rectangle;

    int can_place_rectangle(double *a, Rectangle *placed_rectangles, int num_rectangles,
                            int x, int y, int w, int h, int rect_idx, int W_max, int H_max, int N);
""")



def find_conn_containing_rect(E, rect_idx):
    """
    Filter out all the connections in which a rectangle is involved.

    Parameters:
    ----------
    E (list) : Dictionary of nets with associated cost
    rect_idx (int) : The rectangle index to search for.

    Returns:
    -------
    list : A list of net indices where the rectangle index is found.
    """

    related_conns = []
    
    for idx, (rect_indexes, cost) in enumerate(E):
        if rect_idx in rect_indexes:
            related_conns.append(idx)
    
    return related_conns



def order_symm_group_by_position(G, decoded):
    """
    Order the symmetric group rectangles by their proximity to the origin.

    Parameters:
    ----------
    G (list) : Symmetric group definitions with indices and self symmetry flags.
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.

    Returns:
    -------
    list : Indices of the symmetric group ordered by proximity to the origin.
    """
    
    vec_G = [] # Vectorized G

    for idx in range(len(G)):
        i, j, flag = G[idx]
        if flag == 0:
            d1 = decoded[i][0]**2 + decoded[i][1]**2
            d2 = decoded[j][0]**2 + decoded[j][1]**2
            d_min = min(d1, d2)
        else:
            d_min = decoded[j][0]**2 + decoded[j][1]**2

        # Append the index and the computed distance as a tuple
        vec_G.append((idx, d_min))

    # Sort vec_G by the proximity to the origin
    vec_G.sort(key=lambda x: x[1])
    sorted_indices = [index for index, _ in vec_G]
    
    return sorted_indices



def add_new_points(placed, x, y, w, h, rect_idx):
    """
    Generate new placing points from the corners of a newly placed rectangle and potential intersections with existing rectangles.

    Parameters:
    ----------
    placed (list) : List of rectangles that have already been placed.
    x (float) : X-coordinate of the bottom-left corner of the rectangle's placement.
    y (float) : Y-coordinate of the bottom-left corner of the rectangle's placement.
    w (float) : Width of the rectangle.
    h (float) : Height of the rectangle.
    rect_idx (int) : Index of the rectangle associated to new points.

    Returns:
    -------
    list : A list of tuples representing new placing points.
    """
    
    new_points = [
        (x + w, y, rect_idx),
        (x, y + h, rect_idx),
        (x + w, y + h, rect_idx)
    ]
    new_points += check_obstacle_intersections(placed, x, y, w, h)
    
    return new_points



def check_obstacle_intersections( placed, x, y, w, h):
    """
    Check intersections with obstacles and add new points.

    Parameters:
    ----------
    placed (list) : List of rectangles that have already been placed.
    x (float) : X-coordinate of the bottom-left corner of the rectangle's placement.
    y (float) : Y-coordinate of the bottom-left corner of the rectangle's placement.
    w (float) : Width of the rectangle.
    h (float) : Height of the rectangle.

    Returns:
    -------
    list : new points from intersections with close placed rectangles
    """
    
    new_points = set()
    max_x_left = float('-inf')
    max_y_left = float('-inf')
    max_x_down = float('-inf')
    max_y_down = float('-inf')
    
    for (px, py, pw, ph, pidx, pvar) in placed:
        # Check intersections horizontally 
        if y + h <= py + ph and y + h > py and x <= px + pw and px + pw > max_x_left:
            max_x_left = px + pw
            max_y_left = y + h 
            max_idx_left = pidx
        # Check intersections vertically      
        if x + w <= px + pw and x + w > px and y > py + ph and py + ph > max_y_down:
            max_y_down = py + ph
            max_x_down = x + w 
            max_idx_down = pidx

    # If new points are feasible, add them to placing points
    if max_x_left != float('-inf') and max_y_left != float('-inf'):
        new_points.add((max_x_left, max_y_left, max_idx_left ))
        
    if max_x_down != float('-inf') and max_y_down != float('-inf'):
        new_points.add((max_x_down, max_y_down, max_idx_down))
        
    return list(new_points)



def can_place_rectangle(a, placed, x, y, w, h, rect_idx, W_max, H_max):
    """
    Check if a rectangle can be placed at a given position without overlapping existing rectangles or exceeding boundaries.

    Parameters:
    ----------
    a (list) : Symmetric matrix of minimum distance between rectangles
    placed (list): List of rectangles that have already been placed.
    x (float) : X-coordinate of the bottom-left corner of the rectangle to place.
    y (float) : Y-coordinate of the bottom-left corner of the rectangle to place.
    w (float) : Width of the rectangle to place.
    h (float) : Height of the rectangle to place.
    rect_idx (int) : Index of the rectangle to place.
    W_max (float) : Maximum allowable width of the placement area.
    H_max (float) : Maximum allowable height of the placement area.

    Returns:
    -------
    bool : True if the rectangle can be placed at the given position; False otherwise.
    """
    
    if x < 0 or y < 0 or x + w > W_max or y + h > H_max: # Exceed boundaries?
        return False
        
    for i in range(len(placed) - 1, -1, -1):
        px, py, pw, ph, pidx, pvar =  placed[i]
        if pidx == rect_idx: # Already placed?
            return False
        min_d = a[rect_idx][pidx]
        # Closed already placed rectangles overlapping
        if not (x + w + min_d <= px  or x >= px + pw + min_d or y + h + min_d <= py or y >= py + ph + min_d):
            return False
            
    return True


'''
C ACCELERATED
def can_place_rectangle(a, placed, x, y, w, h, rect_idx, W_max, H_max):
    """
    Call a C function to check if a rectangle can be placed in a given position
    """

    # Load the shared library 
    C = ffi.dlopen('src/librect.so')
    
    a_flat = np.array(a).flatten() if not isinstance(a, np.ndarray) else a.flatten()

    # Cast the flattened array to a C-compatible type
    a_flat_c = ffi.cast("double *", ffi.from_buffer(a_flat))

    # Convert placed to C array of Rectangle structs
    placed_rects_c = ffi.new("Rectangle[]", [(int(px), int(py), int(pw), int(ph), int(pidx), int(pvar)) 
                                             for (px, py, pw, ph, pidx, pvar) in placed])

    result = C.can_place_rectangle(
        a_flat_c,                              # Flattened matrix cast to double pointer
        placed_rects_c,                        # C array of rectangles
        int(len(placed)),           # Number of placed rectangles
        int(x), int(y), int(w), int(h), int(rect_idx),  # Cast all coordinates and properties to integers
        int(W_max), int(H_max),                # Maximum width and height (integers)
        int(len(a))                            # Size of the flattened matrix (number of elements)
    )
    
    return bool(result)
'''



def place_rectangle(a, placed, P, pos, w, h, rect_idx, var, W_max, H_max, min_x, min_y, max_x, max_y):
    """
    Place a rectangle at a given position and update the set of placing points and boundary limits.

    Parameters:
    ----------
    a (list) : Symmetric matrix of minimum distance between rectangles.
    placed (list) : List of rectangles that have already been placed.
    P (list) : List of available placing points.
    pos (tuple) : The position where the new rectangle will be placed.
    w (float) : Width of the rectangle.
    h (float) : Height of the rectangle.
    rect_idx (int) : Index of the rectangle being placed.
    var (int) : Variation or orientation of the rectangle.
    W_max (float) : Maximum allowable width of the placement area.
    H_max (float) : Maximum allowable height of the placement area.
    min_x (float) : Minimum x-coordinate in the placement area.
    min_y (float) : Minimum y-coordinate in the placement area.
    max_x (float) : Maximum x-coordinate in the placement area.
    max_y (float) : Maximum y-coordinate in the placement area.

    Returns:
    -------
    None
    """
    x, y, _ = pos

    # Place the rectangle
    placed.append([x, y, w, h, rect_idx, var])
    
    # Update the set of points P
    new_points = add_new_points(placed, x, y, w, h, rect_idx)
    P.extend(point for point in new_points if point not in P)  # Avoid adding duplicate points

    # Update extreme points of placement
    min_x = min(min_x, x)
    min_y = min(min_y, y)
    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)



def order_decoded_by_position(decoded, indices):
    """
    Sort a subset of rectangles based on the Euclidean distance of their bottom-left corner from the origin (0, 0).

    Parameters:
    ----------
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.
    indices (list) : List of indices representing the subset of rectangles to be sorted.

    Returns:
    list : A list of indices corresponding to the sorted rectangles based on their distance from the origin.
    """
    
    filtered_rectangles = [decoded[i] for i in indices]
    sorted_points = sorted(filtered_rectangles, key=lambda p: p[0]**2 + p[1]**2)
    return [decoded.index(p) for p in sorted_points]


def evaluate_placement( E, E_indices_ptr, E_costs, placed, x, y, w, h, idx, var, cost_conn, cost_area, min_x, min_y, max_x, max_y):
    """
    Evaluate the placement of a rectangle based on area and connectivity criteria.

    Parameters:
    ----------
    E (list) : Dictionary of nets with associated cost
    E_indices_ptr (ffi object) : C param (cffi) containing indices of connection list E
    E_costs (ffi object) : C param (cffi) containing costs of connection list E
    placed (list) : List of rectangles that have already been placed.
    x (float) : X-coordinate of the new rectangle's placement.
    y (float) : Y-coordinate of the new rectangle's placement.
    w (float) : Width of the new rectangle.
    h (float) : Height of the new rectangle.
    idx (int) : Index of the new rectangle.
    var (float) : Variant of the new rectangle .
    cost_conn (float) : Cost coefficient for connectivity.
    cost_area (float) : Cost coefficient for the area.
    min_x (float) : Minimum x-coordinate in the placement area.
    min_y (float) : Minimum y-coordinate in the placement area.
    max_x (float) : Maximum x-coordinate in the placement area.
    max_y (float) : Maximum y-coordinate in the placement area.

    Returns:
    -------
    float: The score of the placement following area and connectivity criteria.
    """
    
    min_width = max(max_x, x + w) - min(min_x, x)
    min_height = max(max_y, y + h) - min(min_y, y)

    # Compute Half-Perimeter Wirelength with C acceleration
    conn_HPWL = deu.conn_HPWL_C(len(E), E_indices_ptr, E_costs, placed + [[x, y, w, h, idx, var]])
    
    return cost_area * (min_width + min_height) + cost_conn * conn_HPWL


def place_symmetry_group(R, G, E, E_indices_ptr, E_costs, E_cache, a, P, decoded, placed_rectangles, pm, W_max, H_max, cost_conn, cost_area, min_x, min_y, max_x, max_y):
    """
    Place rectangles that are part of a symmetric group.

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G (dict) : Symmetric group definitions with indices and self symmetry flags.
    E (list) : Dictionary of nets with associated cost
    E_indices_ptr (ffi object) : C param (cffi) containing indices of connection list E
    E_costs (ffi object) : C param (cffi) containing costs of connection list E
    E_cache (list): Cached connection indeces where rectangles are involved
    a (list) : Symmetric matrix of minimum distance between rectangles
    P (list) : List of available placing points.
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.
    placed (list) : List of rectangles that have already been placed.
    pm (float) : Priority module 
    W_max (float) : Maximum allowable width of the placement area.
    H_max (float) : Maximum allowable height of the placement area.
    cost_conn (float) : Cost coefficient for connectivity.
    cost_area (float) : Cost coefficient for the area.
    min_x (float) : Minimum x-coordinate in the placement area.
    min_y (float) : Minimum y-coordinate in the placement area.
    max_x (float) : Maximum x-coordinate in the placement area.
    max_y (float) : Maximum y-coordinate in the placement area.

    Returns:
    -------
    None
    """
    
    ordered_indices = order_symm_group_by_position(G, decoded)

    # Main idea: assemble symmetric group and place it closer to origin
    # then find a best position and use it as an offet for the assemble
    
    symm_preplaced = []
    xG = 0

    # Find mininum vertical axis of symmetry
    for i, j, flag in G.values():
        if flag == 0:
            xG = max(xG, R[i][decoded[i][2]][0]+a[i][j]/2)
        else:
            xG = max(xG, R[i][decoded[i][2]][0]/2)
            
    h_tmp = 0
    # Preplace symmetric group, considering positional priority
    # and inter-device minimum distances
    for key in ordered_indices:
        i, j, flag = G[key]
        var = decoded[i][2]
        w = R[i][var][0]
        h = R[i][var][1]
        x = xG - (w +a[i][j]/2) if flag == 0 else xG - w / 2
        y = h_tmp
        if key < len(ordered_indices) and h_tmp > 0:
            next_i, next_j, next_flag = G[key]

            # Compute vertical stack minimum distance
            if flag:
                if next_flag:
                    h_tmp += a[i][next_i]
                else:
                    h_tmp += max(a[i][next_i], a[i][next_j])
            else:
                if next_flag:
                    h_tmp += max(a[i][next_i], a[j][next_i])
                else:
                    h_tmp += max(a[i][next_i], a[j][next_j])

        symm_preplaced.append([x, y, w, h, i, var])

        if flag == 0:
            symm_preplaced.append([xG + a[i][j] /2 , h_tmp, w, h, j, var])
            decoded[j] = (decoded[j][0], decoded[j][1], decoded[i][2], decoded[i][3])

        h_tmp += h
    
    best_pos = None
    best_score = float('inf')
    best_delta_x = 0
    best_delta_y = 0

    max_x_symm = max((x + w for x, y, w, h, idx, var in placed_rectangles), default=0)

    # Two extra placing points for avoiding free space pocket
    sec_points = {(symm_preplaced[0][0], 0, None), (max_x_symm + xG - symm_preplaced[0][0], 0, None)}
    P.extend(sec_points - set(P))  # Add only unique points to P

    # Find best position to place entire symmetric group assemble
    for pos in P:
        x_tmp, delta_y, who  = pos
        delta_x = x_tmp - symm_preplaced[0][0]
        score = 0
        flag_place = True
        
        d = 0
        
        if who:
            d = a[symm_preplaced[0][4]][who]

        for i in range(len(symm_preplaced)):
            x, y, w, h, idx, var = symm_preplaced[i]
            new_x = x + delta_x
            new_y = y + delta_y

            d_info = (0,  True)

            # Try to place symmetric group with horizontal spacing to ensure minimum distance with who
            if can_place_rectangle(a, placed_rectangles, new_x + d, new_y, w, h, idx, W_max, H_max): 
                score += evaluate_placement(
                    E, E_indices_ptr, E_costs, placed_rectangles, new_x + d, new_y, w, h, idx, var, cost_conn, cost_area, min_x, min_y, max_x, max_y
                )
                d_info = (d,  True)
            # Try to place symmetric group with vertically spacing to ensure minimum distance with who
            elif can_place_rectangle(a, placed_rectangles, new_x, new_y + d, w, h, idx, W_max, H_max): 
                score += evaluate_placement(
                    E, E_indices_ptr, E_costs, placed_rectangles, new_x, new_y + d, w, h, idx, var, cost_conn, cost_area, min_x, min_y, max_x, max_y
                )
                d_info = (d,  False)
            else:
                flag_place = False
                break
                
        if flag_place and score < best_score:
            best_score = score
            best_pos = pos
            best_delta_x = delta_x
            best_delta_y = delta_y
            best_min_dist = d_info

    # Is best position feasible? Entire symmetric group must be placed
    if best_pos:
        for i in range(len(symm_preplaced)):
            x, y, w, h, idx, var = symm_preplaced[i]
            new_points = set()
            
            if best_min_dist[1]:  # Horizontal spacing
                placed_rectangles.append([x + best_delta_x + best_min_dist[0], y + best_delta_y, w, h, idx, var])
                new_points = add_new_points(placed_rectangles, x + best_delta_x + best_min_dist[0], y + best_delta_y, w, h, idx)
                
                # Update extreme points of placement
                min_x = min(min_x, x + best_delta_x + best_min_dist[0])
                min_y = min(min_y, y + best_delta_y)
                max_x = max(max_x, x + best_delta_x + best_min_dist[0] + w)
                max_y = max(max_y, y + best_delta_y + h)
                
            else:  # Vertical spacing
                placed_rectangles.append([x + best_delta_x, y + best_delta_y + best_min_dist[0], w, h, idx, var])
                new_points = add_new_points(placed_rectangles, x + best_delta_x, y + best_delta_y + best_min_dist[0], w, h, idx)
                
                # Update extreme points of placement
                min_x = min(min_x, x + best_delta_x )
                min_y = min(min_y, y + best_delta_y + best_min_dist[0])
                max_x = max(max_x, x + best_delta_x + w)
                max_y = max(max_y, y + best_delta_y + best_min_dist[0] + h)
                
            P.extend(new_points)

            # If best position is taken, remove it from P
            if best_min_dist[0] <= 0 and best_pos in P:
                P.remove(best_pos)

            # Apply priority-modulation
            if idx in E_cache[idx] :
                for i in E_cache[idx]:
                    for j in E[i][0]:
                        decoded[j] = (decoded[j][0] * pm, decoded[j][1] * pm, decoded[j][2], decoded[i][3])



def place_non_symm_rectangle(R, E, E_indices_ptr, E_costs, E_cache, a, P, decoded, placed, rect_idx, W_max, H_max, pm, cost_conn, cost_area, min_x, min_y, max_x, max_y):
    """
    Place a rectangle that is not part of a symmetric group in the optimal position among available points.

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    E (list) : Dictionary of nets with associated cost
    E_indices_ptr (ffi object) : C param (cffi) containing indices of connection list E
    E_costs (ffi object) : C param (cffi) containing costs of connection list E
    E_cache (list): Cached connection indeces where rectangles are involved
    a (list) : Symmetric matrix of minimum distance between rectangles
    P (list) : List of available placing points.
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.
    placed (list) : List of rectangles that have already been placed.
    rect_idx (int) : Index of the rectangle being placed.
    W_max (float) : Maximum allowable width of the placement area.
    H_max (float) : Maximum allowable height of the placement area.
    pm (float) : Priority module 
    cost_conn (float) : Cost coefficient for connectivity.
    cost_area (float) : Cost coefficient for the area.
    min_x (float) : Minimum x-coordinate in the placement area.
    min_y (float) : Minimum y-coordinate in the placement area.
    max_x (float) : Maximum x-coordinate in the placement area.
    max_y (float) : Maximum y-coordinate in the placement area.

    Returns:
    -------
    None
    """

    # Initialize best position attributes
    best_pos = None
    best_score = float('inf')
    best_min_dist = 0
    best_space_flag = True 

    var = decoded[rect_idx][2]
    w, h = R[rect_idx][var]

    # Find best to position to place the rectangle
    for pos  in P:
        x, y, who = pos
        d = 0
        if who:
             d = a[rect_idx][who]

        # Is position feasible with horizontal spacing? It is needed to ensure minimum distance
        if can_place_rectangle(a, placed, x + d, y, w, h, rect_idx, W_max, H_max):
            score = evaluate_placement(
               E, E_indices_ptr, E_costs, placed, x + d, y, w, h, rect_idx, var, cost_conn, cost_area, min_x, min_y, max_x, max_y
            )
            if score < best_score:
                best_score = score
                best_pos = pos
                best_space_flag = True # Horizontal spacing 
                best_min_dist = d

        # Is position feasible with vertical spacing?
        elif can_place_rectangle(a, placed, x, y + d, w, h, rect_idx, W_max, H_max):
            score = evaluate_placement(
                E, E_indices_ptr, E_costs, placed, x, y + d, w, h, rect_idx, var, cost_conn, cost_area, min_x, min_y, max_x, max_y
            )
            if score < best_score:
                best_score = score
                best_pos = pos
                best_space_flag = False # Vertical spacing 
                best_min_dist = d

    # If best position is feasible, place the rectangle
    if best_pos:
        if best_space_flag and best_min_dist > 0 : # Horizontal spacing
            best_pos = (best_pos[0] + best_min_dist, best_pos[1], best_pos[2])
        elif best_space_flag == False and best_min_dist > 0: # Vertical spacing
            best_pos = (best_pos[0], best_pos[1] + best_min_dist, best_pos[2])
            
        place_rectangle(a, placed, P, best_pos, w, h, rect_idx, var, W_max, H_max, min_x, min_y, max_x, max_y)

        # If best position is taken, remove it from P
        if best_min_dist <= 0 and best_pos in P: 
            P.remove(best_pos)

    # Apply priority modulation with caching
    if rect_idx in E_cache[rect_idx] :
        for i in E_cache[rect_idx]:
            for j in E[i][0]:                  
                    
                decoded[j] = (decoded[j][0] * pm, decoded[j][1] * pm, decoded[j][2], decoded[i][3]) 



def heuristic_placement(R, G_list, E, a, P, decoded, placed, pm, W_max, H_max, cost_conn, cost_area):
    """
    Place rectangles using a constructive heuristic approach.
    
    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    P (list) : List of available placing points.
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.
    placed (list): List of rectangles that have already been placed.
    pm (float) : Priority module 
    W_max (float) : Maximum allowable width of the placement area.
    H_max (float) : Maximum allowable height of the placement area.
    cost_conn (float) : Cost coefficient for connectivity.
    cost_area (float) : Cost coefficient for the area.

    Returns:
    -------
    list :  List of rectangles representing a feasible layout.
    """

    # Prepare the E as C param 
    E_indices = []
    E_costs = ffi.new(f"double[]", [cost for _, cost in E])
    for conn_indices, _ in E:
        E_indices.append(ffi.new("int[]", conn_indices + [-1]))  # Append -1 to mark end of list
    
    E_indices_ptr = ffi.new(f"int*[{len(E)}]", E_indices) # Convert the list of arrays into a pointer array

    # Initialize extreme points of placement
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for (px, py, pw, ph, pidx, pvar) in placed:
        min_x = min(min_x, px)
        min_y = min(min_y, py)
        max_x = max(max_x, px + pw)
        max_y = max(max_y, py + ph)

    # Extract all symmetric group rectangles indices
    vec_G_global = []
    # and extract most priority one per each group
    min_symm_indices = [ 0 for _ in range(len(G_list))]
    
    for k in range(len(G_list)):
    
        vec_G = []
        
        for key in G_list[k]:
            i, j, flag = G_list[k][key]
            vec_G.append(i)
            if flag == 0:
                vec_G.append(j)

        vec_G_global += vec_G
    
        min_value = float('inf')
        
        for idx in vec_G:
            current_value = decoded[idx][0] ** 2 + decoded[idx][1] ** 2
            if current_value < min_value:
                min_value = current_value
                min_symm_indices[k] = idx
                
    # Initialize priority-sorted list of rectangle indices 
    sorted_points = sorted(decoded, key=lambda p: p[0] ** 2 + p[1] ** 2)
    ord_decoded_indices = [decoded.index(p) for p in sorted_points]

    # Caching find_conn_containing_rect results [increase performance]
    E_cache = {i: find_conn_containing_rect(E, i) for i in range(len(decoded))}

    while True:
        rect_idx = ord_decoded_indices[0] # Higher priority rectangle to place

        # Placement of symmetric group rectangles -----------------------------------------------
        
        if rect_idx in min_symm_indices: 
            # First rectangle in a symmetric group trigger placement of this last
            idx = min_symm_indices.index(rect_idx)
            place_symmetry_group(
                R, G_list[idx], E, E_indices_ptr, E_costs, E_cache, a, P, decoded, placed, 
                pm, W_max, H_max, cost_conn, cost_area, min_x, min_y, max_x, max_y
            )
            if len(ord_decoded_indices) > 1: # Update next rectangles to place, due to priority-modulation
                ord_decoded_indices = order_decoded_by_position(decoded,ord_decoded_indices[1:])
                continue
            else:
                break
                
        elif rect_idx in vec_G_global and not rect_idx in min_symm_indices: # Not first rectangle in a symmetric group: skipped 
            if len(ord_decoded_indices) > 1: # Update next rectangles to place, due to priority-modulation
                ord_decoded_indices = order_decoded_by_position(decoded,ord_decoded_indices[1:])
                continue
            else:
                break
        

        # Placement of non-symmetric group rectangles -----------------------------------------------

        place_non_symm_rectangle(R, E, E_indices_ptr, E_costs, E_cache, a, P, decoded, placed, rect_idx, W_max, H_max, pm, cost_conn, cost_area, min_x, min_y, max_x, max_y)
        
        if len(ord_decoded_indices) > 1: # Update next rectangles to place, due to priority-modulation
            ord_decoded_indices = order_decoded_by_position(decoded,ord_decoded_indices[1:])
        else:
            break
            
    return placed

