import random
from random import uniform
import math
import json
from cffi import FFI

ffi = FFI()

# Define the C function signature
ffi.cdef("""
    double conn_HPWL(int E_len, int* E_indices[], double* E_costs, int placed_len, double placed[][6]);
""")

C = ffi.dlopen('src/conn_HPWL.so')



def convert_keys_to_int(data):
    """
    Recursively convert dictionary keys back to integers.
    
    Parameters:
    ----------
    data (dict) : The dictionary with string keys.
    
    Returns:
    -------
    dict : Dictionary with integer keys.
    """
    
    if isinstance(d, dict):
        return {int(k): convert_keys_to_int(v) for k, v in data.items()}
    return data



def save_data_json(filename, R, G_list, E, a):
    """
    Save placement context (R, G_list, E, a) to a JSON file.
    
    Parameters:
    ----------
    filename (string) : The file to save data into.
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles

    Returns:
    -------
    None
    """
    
    with open(filename, 'w') as file:
        data = {
            'R': {str(k): v for k, v in R.items()},  # Convert dictionary keys to strings
            'G_list': {str(k): v for k, v in G_list.items()},  # Convert dictionary keys to strings
            'E': E,
            'a': a
        }
        # Serialize the data to JSON and write to the file
        json.dump(data, file, indent=4)



def load_data_json(filename):
    """
    Load placement context from a JSON file.
    
    Parameters:
    ----------
    filename (string) : The file to load context from.
    
    Returns:
    -------
    Context :  (R, G_list, E, a) as deserialized objects.
    """
    
    with open(filename, 'r') as file:
        # Deserialize the JSON data from the file
        data = json.load(file)
        R = convert_keys_to_int(data['R']) # Convert keys back to integers 
        G_list = convert_keys_to_int(data['G_list']) # Convert keys back to integers
        E = data['E']
        a = data['a']
        return R, G_list, E, a



def enumerate_macro_rectangles(rectangles, index=1, current_width=None, current_height=30, max_width=None, total_height=30, configurations=None):
    """
    Recursively enumerate possible configurations of macro rectangles.
    
    Parameters:
    ----------
    rectangles (list) : List of rectangle widths to arrange.
    index (int) : Current rectangle index.
    current_width (float) : Width of the current configuration.
    current_height (float) : Height of the current configuration.
    max_width (float) : Maximum width encountered so far.
    total_height (float) : Total height of the current configuration.
    configurations (set) : Set of unique configurations.
    
    Returns:
    -------
    set : Set of configurations (max_width, total_height).
    """
    
    if configurations is None:
        configurations = set()
    
    # Initialize with the first rectangle if it's the first call
    if index == 1:
        current_width = rectangles[0]
        max_width = rectangles[0]
    
    n = len(rectangles)
    
    # Base case: if all rectangles have been placed, add the configuration
    if index == n:
        configurations.add((max_width, total_height))
        return
    
    # Get the current rectangle's width and height
    rect_width = rectangles[index]
    rect_height = 30  # Since all rectangles have the same height
    
    # Case 1: Place the rectangle beside the current configuration (horizontally)
    new_width = current_width + rect_width
    new_max_width = max(max_width, new_width)
    enumerate_macro_rectangles(rectangles, index + 1, new_width, current_height, new_max_width, total_height, configurations)
    
    # Case 2: Place the rectangle on top of the current configuration (vertically)
    new_total_height = total_height + rect_height
    enumerate_macro_rectangles(rectangles, index + 1, rect_width, rect_height, max_width, new_total_height, configurations)
    
    return configurations



def init_rectangles_random(N, W_max, H_max, SEED):
    """
    Initialize rectangles randomly with variants and groupings.
    
    Parameters:
    ----------
    N (int) : The total number of rectangles.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    SEED (int) : Random seed for reproducibility.
    
    Returns:
    -------
    list : Dictionary of variants.
    list : List of symmetry groups
    list : Matrix of minimum distance between rectangles
    """

    random.seed(SEED)
    
    R = {}
    G_list = {}
    G = {}
    
    rectangles = {}
    rectangles[0] = [10, 5, 2]
    rectangles[1] = [20, 10, 5, 3, 2]

    for i in range(0, N-6):
        R[i] = {}  # Initialize the dictionary for each rectangle
        R[i][0] = [0, 0]  # Initialize each configuration as a list
        R[i][1] = [0, 0]
        R[i][0][0] = uniform(10, 100)  # width
        R[i][0][1] = uniform(10, 100)  # height
        R[i][1][0] = R[i][0][1]
        R[i][1][1] = R[i][0][0] 
        
    G[0] = [N-6-1, 0, 1]

    # Symmetric groups
    for i in range(N-6, N-2, 2):
        R[i] = {}  # Initialize the dictionary for each rectangle
        R[i][0] = [0, 0]  # Initialize each configuration as a list
        R[i][1] = [0, 0]
        R[i][0][0] = uniform(10, 100)  # width
        R[i][0][1] = uniform(10, 100)  # height
        R[i][1][0] = R[i][0][1]
        R[i][1][1] = R[i][0][0] 

        R[i+1] = {}  # Initialize the dictionary for each rectangle
        R[i+1][0] = [0, 0]  # Initialize each configuration as a list
        R[i+1][1] = [0, 0]
        R[i+1][0][0] = R[i][0][0]  # width
        R[i+1][0][1] = R[i][0][1]  # height
        R[i+1][1][0] = R[i+1][0][1]
        R[i+1][1][1] = R[i+1][0][0] 

        G[int((i-(N-6))/2+1)] = {}
        G[int((i-(N-6))/2+1)] = [i, i+1, 0]

    G_list[0] = G
    
    # Topological structures
    for i in range(N-2, N):

        rectangles[i-(N-2)].sort()
        R[i] = list(enumerate_macro_rectangles(rectangles[i-(N-2)])) 


    # Minimum distance matrix
    a = [[uniform(-1, 10) for _ in range(N)] for _ in range(N)]

    for i in range(N-7, N-2):
        for j in range(N-7, N-2):
            a[i][j] = 0

    for i in range(N):
        for j in range(N):
            a[i][j] = a[j][i]
    
    
    return R, G_list, a



def init_nets_random( N, num_nets, SEED):
    """
    Initialize nets (connections) between rectangles randomly.
    
    Parameters:
    ----------
    N (int) : The total number of rectangles.
    num_nets (int) : Number of nets to create.
    SEED (int) : Random seed for reproducibility.
    
    Returns:
    -------
    list :  List of connections (E), where each connection has a list of devices and a cost.
    """
  
    random.seed(SEED)
    
    E = []
    
    for _ in range(num_nets):
        # Randomly select the number of devices connected by this net
        num_devices_in_net = random.randint(2, int(N/3))
        
        # Randomly select a subset of devices for this net
        connected_devices = random.sample(range(N), num_devices_in_net)
        
        # Assign a random cost greater than 0 (or set to 1)
        cost = random.randint(0, 10)  # You can customize the range of costs
        
        E.append([connected_devices, cost])
        
    return E



def inverse_cantor_pairing(c, W_max, H_max):
    """
    Compute inverse of limited Cantor pairing function to decode positions.
    
    Parameters:
    ----------
    c (float) : Cantor-paired number.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    
    Returns:
    -------
    (int, int) : x, y coordinates decoded from Cantor pairing.
    """
    
    # Calculate the maximum possible Cantor pair value (z_max) for the given W_max and H_max
    z_max = (W_max + H_max) * (W_max + H_max + 1) // 2 + H_max
    
    # Scale c by z_max to get the actual Cantor number z
    z = int(c * z_max)
    
    # Inverse Cantor pairing function logic
    w = math.floor((math.sqrt(8 * z + 1) - 1) / 2)
    t = w * (w + 1) // 2
    y = z - t
    x = w - y

    # Ensure x and y are within bounds of W_max and H_max
    if x < 0 or x > W_max or y < 0 or y > H_max:
        # Adjust x and y to be within valid bounds, if necessary
        x = min(max(0, x), W_max)
        y = min(max(0, y), H_max)
    
    return x, y



def decode_chromosome(R, N, chromosome, W_max, H_max):
    """
    Decode a chromosome into a list of rectangle positions and variants.
    
    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    N (int) : The total number of rectangles.
    chromosome (list): Encoded chromosome with positions, variants, and directions.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    
    Returns:
    -------
    list : List of rectangles with their decoded positions and variants.
    float : Priority module.
    """
    
    # Extract position, variant, and priority modulation genes
    positions = chromosome[0:N]
    variants = chromosome[N:2*N]
    directions = chromosome[2*N:3*N]
    priority_module = chromosome[-1]

    # Create list of rectangles with their associated genes
    rectangles = [ (positions[i], variants[i], directions[i],  i) for i in range(N) ]

    # Decode each rectangle
    decoded_solution = []
    for rect in rectangles:
        x, y  = inverse_cantor_pairing(rect[0], W_max, H_max)
        variant_index = int(rect[1] * len(R[rect[3]]))

        decoded_solution.append((x, y, variant_index, rect[2]))  # Create a tuple with decoded information

    return decoded_solution, priority_module



def find_macrorectangle(placed):
    """
    Calculate the minimum edge lengths of a box that can contain all rectangles.
    
    Parameters:
    ----------
    placed (list) : List of rectangles representing the feasible layout of already placed components.
    
    Returns:
    -------
    int, int : Minimum width and length of the bounding box
    """
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    

    for i in range(len(placed)):
        x = placed[i][0]
        y = placed[i][1]
        
        width = placed[i][2]
        height = placed[i][3]

        # Update minimum x and y (left-bottom of the macrorectangle)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        
        # Update maximum x and y (right-top of the macrorectangle)
        max_x = max(max_x, x + width)
        max_y = max(max_y, y + height)
    
    # Calculate minimum width and height of the macrorectangle
    min_width = max_x - min_x
    min_height = max_y - min_y

    return min_width, min_height



def conn_HPWL_C(E_len, E_indices_ptr, E_costs, placed):
    """
    Compute Half-Perimeter Wirelength with C acceleration
    
    Parameters:
    ----------
    E_len (int): Length of the connection list E.
    E_indices_ptr (cffi object) : C param (cffi) containing indices of connection list E
    E_costs (cffi object) : C param (cffi) containing costs of connection list E
    placed (list) : List of rectangles that have already been placed.
    
    Returns:
    -------
    int : Total HPWL for the given placement.
    """

    # Prepare the placed array to be C param
    placed_len = len(placed)
    placed_array = ffi.new(f"double[{placed_len}][6]", placed)

    # Call the C function
    return C.conn_HPWL(E_len, E_indices_ptr, E_costs, placed_len, placed_array)

    

def conn_HPWL(E, placed ):
    """
    Compute Half-Perimeter Wirelength

    Parameters:
    ----------
    E (list) : Dictionary of nets with associated cost
    placed (list) : List of rectangles that have already been placed.

    Returns:
    -------
    int : Total HPWL for the given placement.
    """
    
    # Create a dictionary to store block positions by their index [Increase computational efficiency]
    position_map = {idx: (x, y, w, h) for x, y, w, h, idx, var in placed}

    total_distance = 0
    total_weight = 0

    if len(placed) > 1: 
        for conn_indices, cost in E:
            if not conn_indices:
                continue
            
            # Initialize bounding box variables
            max_x = float('-inf')
            max_y = float('-inf')
            min_x = float('inf')
            min_y = float('inf')
            
            valid_connection = False  # Track if at least one valid connection exists
            
            # Compute the bounding box for the current connection
            for idx in conn_indices:
                if idx in position_map:
                    x, y, w, h = position_map[idx]
                    
                    # Update min/max values based on the actual bottom-left and top-right corners
                    min_x = min(min_x, x + w/2)
                    min_y = min(min_y, y + h/2)
                    max_x = max(max_x, x + w/2)
                    max_y = max(max_y, y + h/2)
                    
                    valid_connection = True  # Mark as a valid connection
            
            if valid_connection:
                distance = (max_x - min_x) + (max_y - min_y)
                total_distance += cost * distance
                total_weight += cost

    return total_distance / total_weight if total_weight != 0 else 0

