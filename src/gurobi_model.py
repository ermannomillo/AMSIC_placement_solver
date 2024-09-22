import gurobipy as gp
from gurobipy import GRB
import device_util as deu
import IO_util as iou

def AMS_placement_gurobi(R, G_list, E, a, F, X, N, placed, cost_conn, cost_area, cost_prox, cost_face):
    """
    Optimize the placement of rectangles in the layout via Gurobi placement ILP model
    
    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    placed (list) : List of rectangles representing the feasible layout of already placed components.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.

    Returns:
    -------
    list: A list of optimized layout based on Gurobi's model.
    float: The optimal objective value.
    """

    
    l_R = 0.1  # Lower bound for aspect ratio
    u_R = 0.9  # Upper bound for aspect ratio
    M = 1000  # Big M value

    model = gp.Model("AMS Placement")
    model.setParam('outputFlag', 0)  # Avoid verbose output 

    
    #-----------------------------------------------------------------------------------------------
    #    Variables
    #-----------------------------------------------------------------------------------------------
    
    x = model.addVars(N, vtype=GRB.CONTINUOUS, name="x")  # x-coordinate of bottom-left corner
    y = model.addVars(N, vtype=GRB.CONTINUOUS, name="y")  # y-coordinate of bottom-left corner
    w = model.addVars(N, vtype=GRB.CONTINUOUS, name="w")  # Width of rectangle i
    h = model.addVars(N, vtype=GRB.CONTINUOUS, name="h")  # Height of rectangle i
    s = model.addVars({(i, k) for i in range(N) for k in range(len(R[i]))}, vtype=GRB.BINARY, name="s")  # Variant selection [sparse matrices]
    W = model.addVar(vtype=GRB.CONTINUOUS, name="W")  # Total width of layout
    H = model.addVar(vtype=GRB.CONTINUOUS, name="H")  # Total height of layot
    r = model.addVars(N, N, 4, vtype=GRB.BINARY, name="r")  # Spatial relationship between rectangles
    r_R = model.addVar(vtype=GRB.BINARY, name="r_R")  # Binary variable for aspect ratio constraints
    x_G = model.addVars(len(G_list), vtype=GRB.CONTINUOUS, name="x_G")  # Vertical symmetry axis position
    
    # Variables for the maximum and minimum coordinates of each net
    X_M = {}  # X_e^M
    X_m = {}  # X_e^m
    Y_M = {}  # Y_e^M
    Y_m = {}  # Y_e^m

    for e, (indexes, ce) in enumerate(E): 
        X_M[e] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'X_M_{e}')
        X_m[e] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'X_m_{e}')
        Y_M[e] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'Y_M_{e}')
        Y_m[e] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'Y_m_{e}')

    
    #-----------------------------------------------------------------------------------------------
    #    Constraints
    #-----------------------------------------------------------------------------------------------

    # (1) Ensure all rectangles fit within the bounding box
    for i in range(N):
        model.addConstr(x[i] + w[i] <= W, name=f"x_width_bound_{i}")
        model.addConstr(y[i] + h[i] <= H, name=f"y_height_bound_{i}")
    
    # (2) Ensure exactly one variant is selected for each rectangle
    for i in range(N):
        model.addConstr(gp.quicksum(s[i, k] for k in range(len(R[i]))) == 1, name=f"select_one_variant_{i}")
    
    # (3) Define the width and height of each rectangle based on the selected variant
    for i in range(N):
        model.addConstr(w[i] == gp.quicksum(R[i][k][0] * s[i, k] for k in range(len(R[i]))), name=f"width_definition_{i}")
        model.addConstr(h[i] == gp.quicksum(R[i][k][1] * s[i, k] for k in range(len(R[i]))), name=f"height_definition_{i}")
    
    # (4) Ensure that at least one spatial relationship holds between each pair of rectangles
    for i in range(N):
        for j in range(i + 1, N):
            model.addConstr(gp.quicksum(r[i, j, k] for k in range(4)) >= 1, name=f"relationship_{i}_{j}")
    
    # (5), (6), (7), (8) Spatial relationships constraints
    
    for i in range(N):
        for j in range(i + 1, N):
            model.addConstr(x[i] + gp.quicksum(R[i][k][0] * s[i, k] for k in range(len(R[i]))) <= x[j] - a[i][j] + M * (1 - r[i, j, 0]), name=f"left_{i}_{j}")
            model.addConstr(x[j] + gp.quicksum(R[j][k][0] * s[j, k] for k in range(len(R[j]))) <= x[i] - a[i][j] + M * (1 - r[i, j, 1]), name=f"right_{i}_{j}")
            model.addConstr(y[i] + gp.quicksum(R[i][k][1] * s[i, k] for k in range(len(R[i]))) <= y[j] - a[i][j] + M * (1 - r[i, j, 2]), name=f"under_{i}_{j}")
            model.addConstr(y[j] + gp.quicksum(R[j][k][1] * s[j, k] for k in range(len(R[j]))) <= y[i] - a[i][j] + M * (1 - r[i, j, 3]), name=f"over_{i}_{j}")
            
            model.addConstr(r[i,j,0] + r[i,j,2] <= 1, f"redundant1_{i}_{j}")
            model.addConstr(r[i,j,1] + r[i,j,3] <= 1, f"redundant2_{i}_{j}")
     
    for k in range(len(G_list)):
        for key in G_list[k]:
            i, j, flag = G_list[k][key]

            if flag: # Self symmetric
                model.addConstr(2 * x_G[k] == 2 * x[i] + w[i], name=f"symmetry_self_{i}") # (18)
            else: # Symmetric pair
                model.addConstr(2 * x_G[k] == x[j] + x[i] + w[i], name=f"symmetry_{key}") # (14)
                # Same y-coordinate for all symmetry pairs (15)
                model.addConstr(y[i] == y[j], name=f"same_y_{i}_{j}")
                # Same width for all symmetry pairs (16)
                model.addConstr(w[i] == w[j], name=f"same_width_{i}_{j}") 
                # Same height for all symmetry pairs (17)
                model.addConstr(h[i] == h[j], name=f"same_height_{i}_{j}") 

    model.addConstr(l_R * W <= H + M * (1 - r_R), name="aspect_ratio_1") # (20.1)
    model.addConstr(H <= u_R * W + M * (1 - r_R), name="aspect_ratio_2") # (20.2)
    model.addConstr(l_R * H <= W + M * r_R, name="aspect_ratio_3") # (21.1)
    model.addConstr(W <= u_R * H + M * r_R, name="aspect_ratio_4") # (21.2)
    

    for e, (indexes, ce) in enumerate(E):
        # Compute extreme points for each net e
        for i in indexes:
            model.addConstr(X_M[e] >= x[i] + w[i] / 2) # (24)
            model.addConstr(X_m[e] <= x[i] + w[i] / 2) # (25)
            model.addConstr(Y_M[e] >= y[i] + h[i] / 2) # (26)
            model.addConstr(Y_m[e] <= y[i] + h[i] / 2) # (27)

    
    #-----------------------------------------------------------------------------------------------
    #    Objective function
    #-----------------------------------------------------------------------------------------------

    
    # Connectivity criterion
    L_conn = gp.quicksum(E[e][1] * (X_M[e] - X_m[e] + Y_M[e] - Y_m[e]) for e in range(len(E)))

    L_face = 0
    for F_idx in range(len(F)):
        i, face, cost_tmp = F[F_idx]
        if face == 1:
            L_face += cost_tmp * x[i]
        elif face == 2:
            L_face += cost_tmp * (H - (y[i]+h[i]))
        elif face == 3:
            L_face += cost_tmp * (W - (x[i]+w[i]))
        elif face == 4:
            L_face += cost_tmp * y[i] 

    L_prox = 0
    for X_idx in range(len(X)): 
        i, j, cost_tmp = X[X_idx]
        L_prox += cost_tmp * ((x[i]-x[j])**2 + (y[i]-y[j])**2)
    
    
    # Calculate the normalization constants
    S_face = sum(F[idx][2] for idx in range(len(F)))
    S_prox = sum(X[idx][2] for idx in range(len(X)))
    S_conn = sum(E[idx][1] for idx in range(len(E)))
    
    # Objective function: Minimize area and HPWL criteria
    model.setObjective(cost_area * (W + H) + cost_conn * (L_conn / S_conn) + cost_face * (L_face / S_face ) + cost_prox * (L_prox / S_prox ) , GRB.MINIMIZE)

    
    #-----------------------------------------------------------------------------------------------
    #    Set starting feasible solution 
    #-----------------------------------------------------------------------------------------------

    
    # Sort the data based on increasing rectangle index
    sorted_data = sorted(placed, key=lambda item: item[4])
    
    # Extract the x values in the sorted order
    x_sorted = [item[0] for item in sorted_data]
    y_sorted = [item[1] for item in sorted_data]
    w_sorted = [item[2] for item in sorted_data]
    h_sorted = [item[3] for item in sorted_data]
    var_sorted = [item[5] for item in sorted_data]
    
    min_width, min_height = deu.find_macrorectangle(placed)
    
    s_tmp = [[0 for _ in range(len(R[i]))] for i in range(N)]
    
    # Update s_tmp based on var_sorted
    for i in range(N):
        k = var_sorted[i]  # Index to be set to 1
    
        if 0 <= k < len(s_tmp[i]): 
            s_tmp[i][k] = 1
    
    # Set the feasible solution as start values, NOT FIXED
    for i in range(N):
        x[i].start = x_sorted[i]
        y[i].start = y_sorted[i]
        w[i].start = w_sorted[i]
        h[i].start = h_sorted[i]
    
    W.start = min_width
    H.start = min_height

    # Compute symmetry axis for each symmetry group
    for k in range(len(G_list)):
        i, j, flag = G_list[k][0]
        if flag:
            x_G[k].start = w_sorted[i]/2 
        else:
            x_G[k].start = abs((x_sorted[j] - (x_sorted[i] + w_sorted[i]))/2 )

    # Variants are instead fixed
    for i in range(N):
        for k in range(len(R[i])):
            s[i, k].start = s_tmp[i][k]
            s[i, k].lb = s_tmp[i][k]
            s[i, k].ub = s_tmp[i][k]


    # Exclude self symmetric rectangles from fixing
    self_symm_rectangles = []
    
    for k in range(len(G_list)):
        
        for key in G_list[k]:
            i, j, flag = G_list[k][key]
            if flag:
                self_symm_rectangles.append(i)


    # Fix relative position of rectangles
    for i in range(N):
        if i == self_symm_rectangles:
                continue
        for j in range(i + 1, N):
            if i == self_symm_rectangles:
                continue
            
            slack_values = [
                x_sorted[j] - (x_sorted[i] + w_sorted[i]),  # Slack for left constraint
                x_sorted[i] - (x_sorted[j] + w_sorted[j]),  # Slack for right constraint
                y_sorted[j] - (y_sorted[i] + h_sorted[i]),  # Slack for under constraint
                y_sorted[i] - (y_sorted[j] + h_sorted[j])   # Slack for over constraint
            ]

            max_slack_index = slack_values.index(max(slack_values))
    
            # The relative position of two rectangles are determined by maximum slack
            for k in range(4):
                if k == max_slack_index:
                    r[i, j, k].start = 1
                    r[i, j, k].lb = 1
                    r[i, j, k].ub = 1
                else:
                    r[i, j, k].start = 0
                    r[i, j, k].lb = 0
                    r[i, j, k].ub = 0

    #-----------------------------------------------------------------------------------------------
    #    Optimize
    #-----------------------------------------------------------------------------------------------
                    
    time_limit = 600  # Set time limit to 600 seconds
    model.setParam(GRB.Param.TimeLimit, time_limit)

    lp_placed = [(0, 0, 0, 0, 0) for _ in range(N)]
    
    try:
        model.optimize()
        
        # Check if the model has found a feasible solution
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
            for i in range(N):
                lp_placed[i] = (x[i].x, y[i].x, w[i].x, h[i].x, i, var_sorted[i])
            
            min_width, min_height = deu.find_macrorectangle(lp_placed)
    
            # Output results
            print("Minimum width:", min_width)
            print("Minimum height:", min_height)
            print("Objective function value - Gurobi:", model.objVal) 

            if model.status == 2:
                print("Optimal solution")

        else:
            print(f"No feasible solution found, status code: {model.status}")
    
    except gp.GurobiError as e:
        print(f"Error: {e}")
    except AttributeError as e:
        print(f"AttributeError: {e}")

    print("\n")
    return lp_placed
