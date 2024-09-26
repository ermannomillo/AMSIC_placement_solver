# Analog and Mixed-Signal IC Placement Solver

This repository provides a heuristic solution for the placement of Analog and Mixed-Signal Integrated Circuits (AMSICs). The core approach leverages evolutionary algorithms, including Genetic Algorithm (GA) and CMA-ES, to generate initial feasible placement solutions. These solutions are then refined using Integer Linear Programming (ILP) with Gurobi, ensuring enhanced precision.

Additionally, the solver supports multiobjective optimization using NSGA-II, enabling simultaneous optimization of multiple criteria costs. The multiobjective approach can be applied to fitness functions derived from GA or CMA-ES.

To boost computational efficiency, the code leverages multithreading and C-based acceleration for faster execution.


## Requirements

## Usage

The test_parallel.py script allows you to run hyperparameter settings for GA and CMA-ES optimization on a set of rectangles to be placed. Below is the command structure and available options:
Command

    python3 test.py [OPTIONS] N

Where N is the positional argument indicating the number of rectangles to place.

### Options

Genetic Algorithm (GA) Parameters:

    --population_ga POPULATION_GA: Set the population size for GA.
    --generations_ga GENERATIONS_GA: Number of generations for GA.
    --childs CHILDS: Number of children for each GA generation.
    --p_c P_C: Crossover probability for GA.
    --p_m P_M: Mutation probability for GA.
    --elite_size ELITE_SIZE: Number of elite individuals to retain for GA.
    --tournament_size TOURNAMENT_SIZE: Set the tournament size for GA selection.

CMA-ES Parameters:

    --population_cma POPULATION_CMA: Set the population size for CMA-ES.
    --generations_cma GENERATIONS_CMA: Number of generations for CMA-ES.
    --sigma SIGMA: Initial standard deviation for CMA-ES.

Multiobjective hyperparameters tuning:

    --population_nsga POPULATION_NSGA: Set the population size for NSGA-II.
    --generations_nsga GENERATIONS_NSGA: Number of generations for NSGA-II.
    --multiobj MULTIOBJ: Choose metaheuristics to optimize: "CMA" or "GA"

General Parameters:

    --seed SEED: Set a random seed for reproducibility.
    --cost_conn COST_CONN: Define the cost associated with the connection criterion.
    --cost_area COST_AREA: Define the cost associated with the area criterion.
    --cost_prox COST_PROX: Define the cost associated with the proximity criterion.
    --cost_face COST_FACE: Define the cost associated with the interface criterion.
    --json_file JSON_FILE: Specify the JSON file to load R, E, G_list, a, F, and X.

Execution Control:

    --skip_cma: Skip the CMA-ES optimization step.
    --skip_local_search: Skip the local search steps.
    --skip_lp: Skip the Gurobi LP step.

Output and Profiling:

    --plot: Plot the evolution and placement of active pipeline stages.
    --save_performance: Save performance data to a CSV file.
    --profile: Print profiling data of active pipeline stages.

    

### Example

To run the optimization with 50 rectangles using specific population sizes for GA, CMA-ES, and NSGA-II:

    python3 test.py 50 --multiobj CMA --population_ga 50 --generations_ga 100 --cost_area 1 --cost_conn 5 --cost_prox 2 --cost_face 3 --plot


This command places 50 rectangles with the following settings:

- GA population of 50, CMA-ES population of 30, NSGA-II population of 20.
- Runs for 100 GA generations, 50 CMA-ES generations, and 40 NSGA-II CMA-based generations while plotting the results.
    
## Acknowledgements

This project is partially based on the paper:

Josef Grus, Zdeněk Hanzálek, "Automated placement of analog integrated circuits using priority-based constructive heuristic", Computers & Operations Research, Volume 167, 2024, 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
