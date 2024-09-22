import random
import device_util as deu
import con_heuristics as che
import numpy as np
import time
import concurrent.futures


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



class Individual:
    """
    Individual class representing a candidate solution in the genetic algorithm.

    Attributes:
    ----------
    chromosome (list): List of reals [0,1] representing the chromosome (solution).
    context (Context): Context object containing problem parameters.
    fitness (float): Fitness value of the individual.
    """

    def __init__(self, chromosome, context):
        
        self.chromosome = chromosome
        self.context = context
        self.fitness = self.calculate_fitness()

    def calculate_fitness(self):
        """
        Calculate the fitness of the individual based on its chromosome.

        Returns:
        -------
        float: Fitness value of the individual.
        """

        rectangles, pm = deu.decode_chromosome(self.context.R, self.context.N, self.chromosome, self.context.W_max,self.context.H_max)
        
        P = [(0, 0, None)]  
        placed_rectangles = []  
        
        placed = che.heuristic_placement(
        self.context.R, self.context.G_list, self.context.E, self.context.a, self.context.F, self.context.X, P, rectangles, placed_rectangles, pm, self.context.W_max, self.context.H_max, self.context.cost_conn, self.context.cost_area, self.context.cost_prox, self.context.cost_face
        )
        
        min_width, min_height = deu.find_macrorectangle(placed)
        L_conn = deu.conn_HPWL(self.context.E, placed)
        L_face = deu.interface_crit(self.context.F, placed, 0, 0, min_width, min_height)
        L_prox = deu.proximity_crit(self.context.X, placed)

        return self.context.cost_area * (min_width + min_height)+ self.context.cost_conn * L_conn + self.context.cost_face * L_face +  self.context.cost_prox * L_prox if len(placed) == self.context.N  else 100000


def deterministic_tournament_selection(population, T):
    """
    Perform deterministic tournament selection to choose the best individual.

    Parameters:
    ----------
    population (list) : List of individuals in the population.
    T (int) : Tournament size.

    Returns:
    -------
    Individual : The selected individual.
    """
    # Adjust T if it's larger than the population size
    if T > len(population):
        T = len(population)
    
    # Perform tournament selection
    tournament = random.sample(population, T)
    return min(tournament, key=lambda ind: ind.fitness)

def select_parents(population, T):
    """
    Select two parents from the population using tournament selection.

    Parameters:
    ----------
    population (list): List of individuals in the population.
    T (int): Tournament size.

    Returns:
    -------
    tuple : Two selected parents (Individual objects).
    """
    
    p1 = deterministic_tournament_selection(population, T)
    p2 = deterministic_tournament_selection(population, T)
    
    return p1, p2

def crossover(p1, p2, context):
    """
    Perform crossover between two parents to generate a child.

    Parameters:
    ----------
    p1 (Individual): First parent.
    p2 (Individual): Second parent.
    context (Context): Context object containing problem parameters.

    Returns:
    -------
    Individual: The generated child.
    """

   
    point = random.randint(1, len(p1.chromosome) - 1)
    child_chromosome = p1.chromosome[:point] + p2.chromosome[point:]
    
    return Individual(child_chromosome, context)

def mutate(individual, context):
    """
    Perform mutation on an individual.

    Parameters:
    ----------
    individual (Individual): The individual to mutate.
    context (Context): Context object containing problem parameters.

    Returns:
    -------
    Individual: The mutated individual.
    """
    
    chromosome = individual.chromosome[:]
    index = random.randint(0, len(chromosome) - 1)
    chromosome[index] = random.random() 
    
    return Individual(chromosome, context)

def elite_selection(population, elite_size):
    """
    Select the elite individuals from the population.

    Parameters:
    ----------    
    population (list) : List of individuals in the population.
    elite_size (int) : Number of elite individuals to select.

    Returns:
    -------
    list : List of elite individuals.
    """
    
    return sorted(population, key=lambda ind: ind.fitness, reverse=False)[:elite_size]

def generate_child_task(population, p_c, p_m, T,  context):
    """
    Child generation task wrapper

    Parameters:
    ----------
    population (list): List of individuals in the population.
    p_c (float): Crossover probability.
    p_m (float): Mutation probability.
    context (Context): Context object containing problem parameters.

    Returns:
    -------
    Individual: The generated child.
    """
    
    p1, p2 = select_parents(population, T)
    
    if random.random() <= p_c:
        child = crossover(p1, p2, context)
        if random.random() <= p_m:
            child = mutate(child, context)
    else:
        child = mutate(p1, context)
    
    return child

def generate_children_parallel(population, s, p_c, p_m, T, context):
    """
    Generate children in parallel.

    Parameters:
    ----------
    population (list) : List of individuals in the population.
    s (int) : Number of children to generate.
    p_c (float) : Crossover probability.
    p_m (float) : Mutation probability.
    context (Context) : Context object containing problem parameters.

    Returns:
    -------
    list : List of generated children.
    """
    
    L = []
    
    # Use ThreadPoolExecutor to parallelize the child generation
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_child_task, population, p_c, p_m, T, context) for _ in range(s)]
        
        for future in concurrent.futures.as_completed(futures):
            L.append(future.result())
    
    return L


def generate_children(population, s, p_c, p_m, T, context):
    """
    Generate children sequentially.

    Parameters:
    ----------
    population (list) : List of individuals in the population.
    s (int) : Number of children to generate.
    p_c (float) : Crossover probability.
    p_m (float) : Mutation probability.
    context (Context) : Context object containing problem parameters.

    Returns:
    -------
    list : List of generated children.
    """
    
    L = []
    while len(L) < s:
        p1, p2 = select_parents(population, T)
        
        if random.random() <= p_c:
            child = crossover(p1, p2, context)
            if random.random() <= p_m:
                child = mutate(child, context)
        else:
            child = mutate(p1, context)
        
        L.append(child)
    
    return L


def genetic_algorithm(R, G_list, E, a, F, X, N, s, p_c, p_m, elite_size, T, generations, population_size, chromosome_length, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face, profile):
    """
    Run the genetic algorithm to generate and optimize a feasible solution

    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    s (int) : Number of offspring to generate each generation.
    p_c (float) : Crossover probability.
    p_m (float) : Mutation probability.
    elite_size (int) : Number of elite individuals to retain.
    T (int) : Tournament size
    generations (int) : Number of generations to run.
    population_size (int) : Size of the population.
    chromosome_length (int) : Length of the chromosome.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.
    profile (bool): Flag to determine if profiling is enabled.

    Returns:
    -------
    tuple: Contains final population, fitness over time, chromosomes over time.
    """

    context = Context(R, G_list, E, a, F, X, N, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face)

    population = [Individual(np.random.rand(chromosome_length).tolist(), context) for _ in range(population_size)]
    fitness_over_time = []
    chromosomes_over_time = []
    
    for i in range(generations):
        if profile:
            children = generate_children(population, s, p_c, p_m, T, context)
        else:
            children = generate_children_parallel(population, s, p_c, p_m, T, context)
        elite = elite_selection(population, elite_size)
        combined = children + elite
        combined = sorted(combined, key=lambda ind: ind.fitness, reverse=False)
        population = combined[:s]
        
        # Collect fitness and chromosome data
        fitness_over_time.append([ind.fitness for ind in population])
        chromosomes_over_time.append([ind.chromosome for ind in population])

        if i%10 == 0:
            print(f"Generation: {i}, Fitness: {min(population, key=lambda ind: ind.fitness).fitness}" )
    
    return population, fitness_over_time, chromosomes_over_time

def run_ga(R, G_list, E, a, F, X, N, s, p_c, p_m, elite_size, T, generations, population_size, chromosome_length, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face, profile):
    """
    Task wrapper to run GA and elaborates performance / profiling
    
    Parameters:
    ----------
    R (list) : Dictionary  of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    s (int) : Number of offspring to generate each generation.
    p_c (float) : Crossover probability.
    p_m (float) : Mutation probability.
    elite_size (int) : Number of elite individuals to retain.
    T (int) : Tournament size
    generations (int) : Number of generations to run.
    population_size (int) : Size of the population.
    chromosome_length (int) : Length of the chromosome.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    cost_conn (float) : The penalty cost associated with connection length between components.
    cost_area (float) : The penalty cost associated with the total area of the layout.
    cost_prox (float) : The penalty cost associated with distance of proximity constraints.
    cost_face (float) : The penalty cost associated with accessibility of interfaces.
    profile (bool): Flag to determine if profiling is enabled.

    Returns:
    -------
    tuple: Contains the final placement, priority module , fitness value, CPU time, fitness over time, chromosomes over time, and the best chromosome (solution).
    """
    
    start_time = time.process_time()
    final_population, fitness_over_time, chromosomes_over_time = genetic_algorithm(
        R, G_list, E, a, F, X, N, s, p_c, p_m, elite_size, T, generations, population_size, chromosome_length, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face, profile
    )
    
    solution_ga = min(final_population, key=lambda ind: ind.fitness)
    rectangles_ga, pm_ga = deu.decode_chromosome(R, N, solution_ga.chromosome, W_max, H_max)
    placed_ga = che.heuristic_placement(R, G_list, E, a, F, X, [(0, 0, None)], rectangles_ga, [], pm_ga, W_max, H_max, cost_conn, cost_area, cost_prox, cost_face)
    fitness_ga = solution_ga.fitness

    end_time = time.process_time()
    ga_cpu_time = end_time - start_time

    print(f"GA Best Fitness: {fitness_ga}")
    print("\n")
    
    return placed_ga, pm_ga, fitness_ga, ga_cpu_time, fitness_over_time, chromosomes_over_time, solution_ga.chromosome
