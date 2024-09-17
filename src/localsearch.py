import con_heuristics as che
import device_util as deu

def local_search_sequence(R, G_list, E, a, N, placed, fitness, pm, W_max, H_max, cost_conn, cost_area):
    """
    Performs a local search over sequences.

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    N (int) : The total number of rectangles to place.
    placed (list) : List of rectangles representing the feasible layout of already placed components.
    fitness (float) : The fitness value associated with the feasible layout (lower is better).
    pm (float) :  The priority module.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.

    Returns:
    -------
    list : The layout of components after performing the local search.
    float : The updated fitness value after local search.
    list : The updated list of rectangle configurations after local search.
    """
    
    best_fitness = fitness
    best_placed = placed

    rectangles = [(0, 0, 0, 0) for _ in range(N)]

    
    for ui in range(N):
        rectangles[best_placed[ui][4]] = (best_placed[ui][0], best_placed[ui][1], best_placed[ui][5], 0)

    for i in range(N):
        # For each rectangle and for each of its variant, try to recompute the constructive heuristics
        for k in range(len(R[i])):
            tmp_var = rectangles[i][2]
            rectangles[i] = (rectangles[i][0], rectangles[i][1], k, 0)
            
            tmp_placed = che.heuristic_placement(R, G_list, E, a, [(0, 0, None )], rectangles, [], pm, W_max, H_max,cost_conn, cost_area)

            min_width, min_height = deu.find_macrorectangle(tmp_placed)
            tmp_fitness = cost_area * (min_width + min_height) + cost_conn * deu.conn_HPWL(E, tmp_placed) if len(tmp_placed) == N else 100000

            if tmp_fitness < best_fitness:
                best_placed = tmp_placed
                best_fitness = tmp_fitness
            else:
                rectangles[i] = (rectangles[i][0], rectangles[i][1], tmp_var, 0)

            for ui in range(N):
                rectangles[best_placed[ui][4]] = (best_placed[ui][0], best_placed[ui][1], rectangles[best_placed[ui][4]][2], 0)

    print(f"Meta fitness: {fitness}, Local search sequence Fitness: {best_fitness}\n")
    
    return best_placed, best_fitness, rectangles

def local_search_layout(R, G_list, E, a, N, placed, decoded, fitness, pm, W_max, H_max, cost_conn, cost_area):
    """
    Performs a local search over layout.

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    N (int) : The total number of rectangles to place.
    placed (list) : List of rectangles representing the feasible layout of already placed components.
    decoded (list) : List of rectangles (chromosomes decoded) that have not yet been placed in a feasible layout.
    fitness (float) : The fitness value associated with the feasible layout (lower is better).
    pm (float) :  The priority module.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.

    Returns:
    -------
    list : The layout of components after performing the local search.
    float : The updated fitness value after local search.
    """

    best_fitness = fitness
    best_placed = placed

    for ui in range(N):
        decoded[best_placed[ui][4]] = (best_placed[ui][0], best_placed[ui][1], decoded[best_placed[ui][4]][2], 0)

    for i in range(N):
        for k in range(len(R[i])):

            tmp_placed = [rect for rect in best_placed if rect[4] != i]
            corners = []

            for rect in tmp_placed:
                x, y, width, height, idx, var = rect
                # Bottom-right corner
                corners.append((x + width, y, idx))
                # Top-left corner
                corners.append((x, y + height, idx))
                # Top-right corner
                corners.append((x + width, y + height, idx))

            tmp_var = decoded[i][2]
            decoded[i] = (decoded[i][0], decoded[i][1], k, 0)
            new_placed = che.heuristic_placement(R, G_list, E, a, corners, decoded, tmp_placed, pm, W_max, H_max,cost_conn, cost_area) 

            min_width, min_height = deu.find_macrorectangle(new_placed)
            tmp_fitness = cost_area * (min_width + min_height) + cost_conn * deu.conn_HPWL(E, new_placed) if len(new_placed) == N else 100000

            if tmp_fitness < best_fitness:
                best_placed = new_placed
                best_fitness = tmp_fitness
            else:
                decoded[i] = (decoded[i][0], decoded[i][1], tmp_var, 0)

            for ui in range(N):
                decoded[best_placed[ui][4]] = (best_placed[ui][0], best_placed[ui][1], decoded[best_placed[ui][4]][2], 0)

    print(f"Sequence local search fitness: {fitness}, Layout local search fitness: {best_fitness}\n")

    return best_placed, best_fitness
