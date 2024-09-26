import numpy as np
import random
from deap import base, creator, tools
import cov_metaheuristics as cov
import gen_metaheuristics as ga
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import IO_util as iou


creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))  # Minimization
creator.create("Individual", list, fitness=creator.FitnessMin)

class Context:
    """
    Context class for holding the parameters used in the genetic algorithm.

    Attributes:
    ----------
    R (list) : Dictionary of available variant per each device or topological structures
    G_list (list) : The list of symmetric groups.
    E (list) : Dictionary of nets with associated cost
    a (list) : Symmetric matrix of minimum distance between rectangles
    F (list) : List of interface rectangles with the associated side to place
    X (list) : List of proximity bounded rectangles 
    N (int) : The total number of rectangles to place.
    W_max (float) : The maximum allowable width for the layout.
    H_max (float) : The maximum allowable height for the layout.
    population_cma (int) : The size of the population for CMA.
    chromosome_length (int) : The length of the chromosomes.
    initial_mean (float) : The initial mean for the population.
    generations (int) : The number of generations to evolve.
    sigma (float) : The standard deviation for mutation.
    s (int) : Number of offspring to generate each generation.
    p_c (float) : Crossover probability.
    p_m (float) : Mutation probability.
    elite_size (int) : Number of elite individuals to retain.
    T (int) : Tournament size
    """
    def __init__(self, R, G_list, E, a, F, X, N, W_max, H_max, population, chromosome_length, initial_mean, generations, sigma, s, p_c, p_m, elite_size, T):
        self.R = R
        self.G_list = G_list
        self.E = E
        self.a = a
        self.F = F
        self.X = X
        self.N = N
        self.W_max = W_max
        self.H_max = H_max
        self.population = population
        self.chromosome_length = chromosome_length
        self.initial_mean = initial_mean
        self.generations = generations
        self.sigma = sigma
        self.s = s
        self.p_c = p_c
        self.p_m = p_m
        self.elite_size = elite_size
        self.T = T



def min_max_scale(values):
    """
    Apply min max scale to a list of values

    Parameters:
    ----------
    values (list) : list of integers
    
    Parameters:
    ----------
    list : Min max scaled values
    
    """
    
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:  # Avoid division by zero
        return [1 for _ in values]
    return [(v - min_val) / (max_val - min_val) for v in values]


def normalize_fitness(population):
    """
    Normalize fitness of the entire current population
    
    Parameters:
    ----------
    population (list) : List of list of criteria costs
    
    """
    
    # Extract fitness values for each objective
    L_area_vals = [ind.fitness.values[0] for ind in population]
    L_conn_vals = [ind.fitness.values[1] for ind in population]
    L_prox_vals = [ind.fitness.values[2] for ind in population]
    L_face_vals = [ind.fitness.values[3] for ind in population]

    # Apply min-max scaling to each objective
    L_area_scaled = min_max_scale(L_area_vals)
    L_conn_scaled = min_max_scale(L_conn_vals)
    L_prox_scaled = min_max_scale(L_prox_vals)
    L_face_scaled = min_max_scale(L_face_vals)

    # Assign scaled fitness back to individuals
    for i, ind in enumerate(population):
        ind.fitness.values = (L_area_scaled[i], L_conn_scaled[i], L_prox_scaled[i], L_face_scaled[i])



def clamp_individual(individual, lower_bound=0, upper_bound=10):
    """
    Clamps each gene of an individual between the specified lower and upper bounds.
    
    Parameters:
    ----------
    individual (list) : List of criteria costs
    lower_bound (int) : Lower bound
    upper_bound (int) : Upper bound
    
    """
    for i in range(len(individual)):
        individual[i] = max(lower_bound, min(upper_bound, individual[i]))


        
def evaluate(individual, context):
    """
    Calculate the fitness of the individual based on its context.

    Parameters:
    ----------
    individual (list) : List of criteria costs
    context (Context) : Support variables

    Returns:
    -------
    tuple : Multicriteria fitness value of the individual.
    """
    
    cost_conn = individual[0]
    cost_area = individual[1]
    cost_prox = individual[2]
    cost_face = individual[3]

    if context.sigma != 0:
        # Optimize CMA
        return cov.run_cma_multiobj(
            context.R, context.G_list, context.E, context.a, context.F, context.X,
            context.N, context.W_max, context.H_max, context.population,
            context.chromosome_length, context.initial_mean, context.generations,
            context.sigma, cost_conn, cost_area, cost_prox, cost_face
        )
    # Optimize GA
    return ga.run_ga_multiobj(
            context.R, context.G_list, context.E, context.a, context.F, context.X,
            context.N, context.s, context.p_c, context.p_m, context.elite_size, context.T,  context.generations, context.population,
            context.chromosome_length, context.W_max, context.H_max,cost_conn, cost_area, cost_prox, cost_face
    )

# NSGA setup
toolbox = base.Toolbox()
toolbox.register("cost_conn", random.uniform, 0, 10)
toolbox.register("cost_area", random.uniform, 0, 10)
toolbox.register("cost_prox", random.uniform, 0, 10)
toolbox.register("cost_face", random.uniform, 0, 10)
toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.cost_conn, toolbox.cost_area, toolbox.cost_prox, toolbox.cost_face), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=10, indpb=0.2)  # Mutation
toolbox.register("select", tools.selNSGA2)  # Selection


def run_nsga2(R, G_list, E, a, F, X, N, W_max, H_max, population_meta, chromosome_length, generations_meta, initial_mean=0, sigma=0,  s=0, p_c=0, p_m=0, elite_size=0, T=0, population_size=20, generations=20):

    """
    Run NSGA-II to multiobjective optimize criteria cost for CMA-ES or GA

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
    population_meta (int) : Size of the metaheuristics population .
    chromosome_length (int) : Length of the chromosome.
    generations_meta (int) : Number of metaheuristics generations to run.
    initial_mean (list): Initial mean vector for the CMA-ES.
    sigma (float): Step size parameter for CMA-ES.
    s (int) : Number of offspring to generate each generation for GA.
    p_c (float) : Crossover probability for GA.
    p_m (float) : Mutation probability for GA.
    elite_size (int) : Number of elite individuals to retain for GA.
    T (int) : Tournament size for GA
    population_size (int) : Size of the NSGA population .
    generations (int) : Number of NSGA generations to run.

    Returns:
    -------
    list : final population

    """

    
    context = Context(R, G_list, E, a, F, X, N, W_max, H_max, population_meta, chromosome_length, initial_mean, generations_meta, sigma, s, p_c, p_m, elite_size, T)

    # Create the population
    population = toolbox.population(n=population_size)
    fitness_over_time = []

    for gen in range(generations):
        # Evaluate the individuals
        fitnesses = list(map(lambda ind: evaluate(ind, context), population))

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Normalize fitness of the population
        #normalize_fitness(population)

        best_individual = tools.selBest(population, 1)[0]
        fitness_over_time.append(best_individual.fitness.values)
        
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # Crossover probability
                toolbox.mate(child1, child2)
                clamp_individual(child1)  # Clamp childs within bounds [0, 10]
                clamp_individual(child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:  # Mutation probability
                toolbox.mutate(mutant)
                clamp_individual(mutant) 
                del mutant.fitness.values

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(lambda ind: evaluate(ind, context), invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Normalize fitness of the offspring
        #normalize_fitness(offspring)

        # Replace the old population by the offspring
        population[:] = offspring

        fits = [ind.fitness.values for ind in population]
        length = len(population)
        mean = sum(map(sum, fits)) / (length * len(fits[0]))
        sum2 = sum(sum(fit) ** 2 for fit in fits)
        std = abs(sum2 / (length * len(fits[0])) - mean ** 2) ** 0.5

        print(f"Generation: {gen}")
        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {mean}")
        print(f"  Std {std}")

    return population, fitness_over_time



def plot_objectives(population):
    """
    Plots the objectives of the population in 3D.
    
    Parameters:
    ----------
    population (list) : The population of individuals after optimization.
   
    """
    # Extract fitness values from the population
    objectives = [ind.fitness.values for ind in population if ind.fitness.valid]
    objectives = np.array(objectives)

    plt.close()

    # Check if there are four objectives
    if objectives.shape[1] == 4:

        fig = plt.figure(figsize=(12, 12))

        # Plot Cost Connection vs Cost Area vs Cost Proximity (3D)
        ax = fig.add_subplot(221, projection='3d')
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c='r')
        ax.set_xlabel('Cost Connection')
        ax.set_ylabel('Cost Area')
        ax.set_zlabel('Cost Proximity')
        ax.set_title('Cost Connection vs Cost Area vs Cost Proximity')

        # Plot Cost Connection vs Cost Area vs Cost Face (3D)
        ax = fig.add_subplot(222, projection='3d')
        ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 3], c='g')
        ax.set_xlabel('Cost Connection')
        ax.set_ylabel('Cost Area')
        ax.set_zlabel('Cost Face')
        ax.set_title('Cost Connection vs Cost Area vs Cost Face')

        # Plot Cost Connection vs Cost Proximity vs Cost Face (3D)
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter(objectives[:, 0], objectives[:, 2], objectives[:, 3], c='b')
        ax.set_xlabel('Cost Connection')
        ax.set_ylabel('Cost Proximity')
        ax.set_zlabel('Cost Face')
        ax.set_title('Cost Connection vs Cost Proximity vs Cost Face')

        # Plot Cost Area vs Cost Proximity vs Cost Face (3D)
        ax = fig.add_subplot(224, projection='3d')
        ax.scatter(objectives[:, 1], objectives[:, 2], objectives[:, 3], c='m')
        ax.set_xlabel('Cost Area')
        ax.set_ylabel('Cost Proximity')
        ax.set_zlabel('Cost Face')
        ax.set_title('Cost Area vs Cost Proximity vs Cost Face')

        plt.tight_layout() 
        plt.show()
        plt.close()  



def plot_convergence(fitness_over_time):
    """
    Plot convergence of criteria
    
    Parameters:
    ----------
    fitness_over_time (list) : List of criteria fitness over time.
        
    """
    generations = range(len(fitness_over_time))
    
    # Separate the fitness values for each objective
    L_area_over_time = [fit[0] for fit in fitness_over_time]
    L_conn_over_time = [fit[1] for fit in fitness_over_time]
    L_prox_over_time = [fit[2] for fit in fitness_over_time]
    L_face_over_time = [fit[3] for fit in fitness_over_time]

    # Plot the convergence of each objective
    plt.figure(figsize=(10, 6))

    plt.scatter(generations, L_area_over_time, label='L_area', marker='o')
    plt.scatter(generations, L_conn_over_time, label='L_conn', marker='x')
    plt.scatter(generations, L_prox_over_time, label='L_prox', marker='s')
    plt.scatter(generations, L_face_over_time, label='L_face', marker='d')

    # Add titles and labels
    plt.xlabel('Generations')
    plt.ylabel('Objective Value')
    plt.title('Convergence of Objectives over Generations')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    plt.show()



def pareto_front(population):
    """
    Extracts the non-dominated individuals from the population.
    
    Parameters:
    ----------
    population (list) : The population of individuals after optimization.
        
    Returns:
    -------
    list : The Pareto front individuals.
    """
    pareto_front = []
    
    for ind in population:
        dominated = False
        for other in population:
            # Check if "other" dominates "ind"
            if np.all(other.fitness.values <= ind.fitness.values) and np.any(other.fitness.values < ind.fitness.values):
                dominated = True
                break  # If dominated, no need to check further
        if not dominated:
            pareto_front.append(ind)
    
    return pareto_front



def plot_pareto_front(pareto_front):

    """
    Plots the Pareto front objectives.
    
    Parameters:
    ----------
    pareto_front : list
        The Pareto front individuals.
    """
    
    # Extract objectives from the Pareto front
    L_area = [ind.fitness.values[0] for ind in pareto_front]
    L_conn = [ind.fitness.values[1] for ind in pareto_front]
    L_prox = [ind.fitness.values[2] for ind in pareto_front]
    L_face = [ind.fitness.values[3] for ind in pareto_front]

    # Create 2D plots for different objective pairs
    plt.figure(figsize=(10, 10))
    
    # L_area vs L_conn
    plt.subplot(221)
    plt.scatter(L_area, L_conn, c='blue')
    plt.xlabel('L_area')
    plt.ylabel('L_conn')
    plt.title('L_area vs L_conn')

    # L_area vs L_prox
    plt.subplot(222)
    plt.scatter(L_area, L_prox, c='green')
    plt.xlabel('L_area')
    plt.ylabel('L_prox')
    plt.title('L_area vs L_prox')

    # L_conn vs L_face
    plt.subplot(223)
    plt.scatter(L_conn, L_face, c='red')
    plt.xlabel('L_conn')
    plt.ylabel('L_face')
    plt.title('L_conn vs L_face')

    # L_prox vs L_face
    plt.subplot(224)
    plt.scatter(L_prox, L_face, c='purple')
    plt.xlabel('L_prox')
    plt.ylabel('L_face')
    plt.title('L_prox vs L_face')

    plt.tight_layout()
    plt.show()
    plt.close() 


