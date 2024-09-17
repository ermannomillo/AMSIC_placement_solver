# Analog and Mixed-Signal IC Placement Solver

This repository implements a heuristic approach for solving the placement of Analog and Mixed-Signal Integrated Circuits (AMSICs). The solver is driven by evolutionary algorithms, specifically Genetic Algorithm (GA) and CMA-ES, which generate feasible placement solutions. These initial solutions are then refined using Integer Linear Programming (ILP), implemented with Gurobi for enhanced precision.

To boost computational efficiency, the code leverages multithreading and C-based acceleration for faster execution.


## Requirements

## Usage

## Acknowledgements

This project is based on the paper:

Josef Grus, Zdeněk Hanzálek, "Automated placement of analog integrated circuits using priority-based constructive heuristic", Computers & Operations Research, Volume 167, 2024, 

## License

This project is licensed under the MIT License - see the LICENSE file for details.
