# Analog and Mixed-Signal IC Placement Solver

This repository implements a heuristic approach for solving the placement of Analog and Mixed-Signal Integrated Circuits (AMSICs). The solver is driven by evolutionary algorithms, specifically Genetic Algorithm (GA) and CMA-ES, which generate feasible placement solutions. These initial solutions are then refined using Integer Linear Programming (ILP), implemented with Gurobi for enhanced precision.

To boost computational efficiency, the code leverages multithreading and C-based acceleration for faster execution.


## Requirements

## Usage

The test_parallel.py script allows you to run hyperparameter settings for GA and CMA-ES optimization on a set of rectangles to be placed. Below is the command structure and available options:
Command

bash

python3 test_parallel.py [OPTIONS] N

Where N is the positional argument indicating the number of rectangles to place.
Options

    -h, --help: Show the help message and exit.

Genetic Algorithm (GA) Parameters:

    --population_ga POPULATION_GA: Set the population size for GA.
    --generations_ga GENERATIONS_GA: Number of generations for GA.
    --childs CHILDS: Number of children for each GA generation.
    --p_c P_C: Crossover probability for GA.
    --p_m P_M: Mutation probability for GA.
    --elite_size ELITE_SIZE: Number of elite individuals to retain for GA.
    --tournament_size TOURNAMENT_SIZE: Set the tournament size for selection in GA.

CMA-ES Parameters:

    --population_cma POPULATION_CMA: Set the population size for CMA-ES.
    --generations_cma GENERATIONS_CMA: Number of generations for CMA-ES.
    --sigma SIGMA: Initial standard deviation for CMA-ES.

General Parameters:

    --seed SEED: Set a random seed for reproducibility.
    --cost_conn COST_CONN: Define the cost associated with the connection criterion.
    --cost_area COST_AREA: Define the cost associated with the area criterion.
    --json_file JSON_FILE: Specify the JSON file to load R, E, G_list, and related data.

Execution Control:

    --skip_cma: Skip the CMA-ES optimization step.
    --skip_local_search: Skip the local search steps.
    --skip_lp: Skip the Gurobi LP step.

Output and Profiling:

    --plot: Plot the evolution and placement of active pipeline stages.
    --save_performance: Save performance data to a CSV file.
    --profile: Print profiling data of active pipeline stages.

Example

To run the optimization with 100 rectangles, using specific population sizes for GA and CMA-ES:



    python3 test_parallel.py --population_ga 50 --population_cma 30 --generations_ga 100 --generations_cma 50 --plot 100

This will place 100 rectangles, using a GA population of 50, a CMA-ES population of 30, and will run for 100 GA generations and 50 CMA-ES generations, while plotting the results.

## Acknowledgements

This project is based on the paper:

Josef Grus, Zdeněk Hanzálek, "Automated placement of analog integrated circuits using priority-based constructive heuristic", Computers & Operations Research, Volume 167, 2024, 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
