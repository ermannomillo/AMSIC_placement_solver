import cma
import time
import numpy as np
import device_util as deu
import con_heuristics as che
from concurrent.futures import ThreadPoolExecutor


class Context:
    """
    Context class for holding the parameters used in the genetic algorithm.

    Attributes:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.
    """
    def __init__(self, R, G_list, E, a, F, X, N, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face):
        self.R = R
        self.G_list = G_list
        self.E = E
        self.a = a
        self.F = F
        self.X = X
        self.N = N
        self.W_max = W_max
        self.H_max = H_max
        self.cost_conn = cost_conn
        self.cost_area = cost_area
        self.cost_prox = cost_prox
        self.cost_face = cost_face

# Define the fitness function
def fitness_function(chromosome, context):
    """
    Evaluate the fitness of a candidate solution.

    Parameters:
    ----------
    chromosome (list): List representing the chromosome (solution).
    context (Context): Context object containing problem parameters.

    Returns:
    -------
    float: Fitness value of the chromosome.
    """
    
    rectangles, pm = deu.decode_chromosome(context.R, context.N, chromosome, context.W_max, context.H_max)

    P = [(0, 0, None)]
    placed_rectangles = [] 
    
    placed = che.heuristic_placement(
        context.R, context.G_list, context.E, context.a, context.F, context.X, P, rectangles, placed_rectangles, pm, context.W_max, context.H_max, context.cost_conn, context.cost_area,  context.cost_prox, context.cost_face
    )

    min_width, min_height = deu.find_macrorectangle(placed)
    L_conn = deu.conn_HPWL(context.E, placed)
    L_prox = deu.proximity_crit(context.X, placed)
    L_face = deu.interface_crit(context.F, placed, 0, 0, min_width, min_height)

    return context.cost_area * (min_width + min_height) + context.cost_conn * L_conn  + context.cost_face * L_face +  context.cost_prox * L_prox if len(placed) == context.N else 100000


def CMA_algorithm(R, G_list, E, a, F, X, N, W_max, H_max, population_size, chromosome_length, initial_mean, generations, sigma, cost_conn, cost_area, cost_prox, cost_face,  profile, stats=True):
    """
    Run the CMA-ES to generate and optimize a feasible solution

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    population_size (int): Size of the population in CMA-ES.
    chromosome_length (int): Length of the chromosome.
    initial_mean (list): Initial mean vector for the CMA-ES.
    generations (int): Number of generations to run.
    sigma (float): Step size parameter for CMA-ES.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.
    profile (bool): Flag to determine if profiling is enabled.
    stats (bool) : Flag to determine if plot evolution stats

    Returns:
    -------
    tuple : Contains the CMA-ES object, fitness values over time, and chromosomes over time.
    """
    
    # Initialize the CMA-ES optimizer
    es = cma.CMAEvolutionStrategy(initial_mean, sigma, {'popsize': population_size})

    fitness_over_time = []
    chromosomes_over_time = []
    context = Context(R, G_list, E, a, F, X, N, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face)
    
    # Run the optimization loop
    for generation in range(generations):
        # Ask for a new population of candidate solutions
        population = es.ask()
        population = [np.clip(individual, 0.01, 0.99) for individual in population]

        if profile:
            fitness_values = [fitness_function(individual, context) for individual in population]
        else:
            # Limit the thread pool to a maximum of 4 threads
            with ThreadPoolExecutor(max_workers=4) as executor:
                fitness_values = list(executor.map(lambda ind: fitness_function(ind, context), population))
        
        # Tell the optimizer the fitness values
        es.tell(population, fitness_values)
        
        # Collect data for analysis
        fitness_over_time.append(fitness_values)
        chromosomes_over_time.append(population)

        if generation%10 == 0 and stats:
            print(f"Generation {generation}, Best Fitness: {min(fitness_values)}")
    
    return es, fitness_over_time, chromosomes_over_time


def run_cma(R, G_list, E, a, F, X, N, W_max, H_max, population_size, chromosome_length, initial_mean, generations, sigma, cost_conn, cost_area, cost_prox, cost_face, profile):
    """
    Task wrapper to run the CMA-ES algorithm and compute its performance / profiling.

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    population_size (int): Size of the population in CMA-ES.
    chromosome_length (int): Length of the chromosome.
    initial_mean (list): Initial mean vector for the CMA-ES.
    generations (int): Number of generations to run.
    sigma (float): Step size parameter for CMA-ES.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.
    profile (bool): Flag to determine if profiling is enabled.

    Returns:
    -------
    tuple: Contains the final placement, priority module, fitness value, CPU time, fitness over time, chromosomes over time, and the best chromosome.
    """
    
    start_time = time.process_time()

    # Ensure the algorithm runs only once
    es, fitness_over_time, chromosomes_over_time = CMA_algorithm(
        R, G_list, E, a, F, X, N, W_max, H_max, population_size, chromosome_length, initial_mean, generations, sigma, cost_conn, cost_area,  cost_prox, cost_face, profile
    )
    
    # Get the best solution and decode it
    solution_cma = es.result.xbest
    fitness_cma = es.result.fbest
    rectangles_cma, pm_cma = deu.decode_chromosome(R, N, solution_cma, W_max, H_max)
    placed_cma = che.heuristic_placement(R, G_list, E, a, F, X, [(0, 0,None)], rectangles_cma, [], pm_cma, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face)

    # Time the execution
    end_time = time.process_time()
    cma_cpu_time = end_time - start_time

    print(f"CMA Best Fitness: {fitness_cma}\n")
    
    return placed_cma, pm_cma, fitness_cma, cma_cpu_time, fitness_over_time, chromosomes_over_time
    


def run_cma_multiobj(R, G_list, E, a, F, X, N, W_max, H_max, population_size, chromosome_length, initial_mean, generations, sigma, cost_conn, cost_area, cost_prox, cost_face):
    """
    Task wrapper to run the CMA-ES algorithm and return non-weighted best multicriteria fitness

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    population_size (int): Size of the population in CMA-ES.
    chromosome_length (int): Length of the chromosome.
    initial_mean (list): Initial mean vector for the CMA-ES.
    generations (int): Number of generations to run.
    sigma (float): Step size parameter for CMA-ES.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.

    Returns:
    -------
    tuple: Contains the L_area, L_conn, L_prox, L_face
    """
    
    
    es, fitness_over_time, chromosomes_over_time = CMA_algorithm(
        R, G_list, E, a, F, X, N, W_max, H_max, population_size, chromosome_length, initial_mean, generations, sigma, cost_conn, cost_area,  cost_prox, cost_face, False, stats=False
    )
    
    # Get the best solution and decode it
    solution_cma = es.result.xbest
    rectangles_cma, pm_cma = deu.decode_chromosome(R, N, solution_cma, W_max, H_max)
    placed_cma = che.heuristic_placement(R, G_list, E, a, F, X, [(0, 0,None)], rectangles_cma, [], pm_cma, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face)

    min_width, min_height = deu.find_macrorectangle(placed_cma)
    
    L_area = min_width + min_height if len(placed_cma) == N else 100000
    L_conn = deu.conn_HPWL(E, placed_cma) if len(placed_cma) == N else 100000
    L_prox = deu.proximity_crit(X, placed_cma) if len(placed_cma) == N else 100000
    L_face = deu.interface_crit(F, placed_cma, 0, 0, min_width, min_height) if len(placed_cma) == N else 100000

    return L_area, L_conn, L_prox, L_face

    
    
