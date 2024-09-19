import sys
import random
import numpy as np
import datetime as dt
import time
import argparse
import cProfile
import pstats
import os

src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import device_util as deu
import con_heuristics as che
import gen_metaheuristics as ga
import cov_metaheuristics_parallel as cov
import localsearch as ls
import gurobi_model as lp
import IO_util as iou



def parse_arguments():
    """Parse console arguments."""
    
    parser = argparse.ArgumentParser(description="Hyperparameter settings for GA and CMA-ES optimization")
    parser.add_argument('N', type=int, default=30, help='Number of rectangles to place')
    parser.add_argument('--population_ga', type=int, default=100, help='Population size for GA')
    parser.add_argument('--population_cma', type=int, default=100, help='Population size for CMA-ES')
    parser.add_argument('--generations_ga', type=int, default=20, help='Number of generations for GA')
    parser.add_argument('--generations_cma', type=int, default=20, help='Number of generations for CMA-ES')
    parser.add_argument('--childs', type=int, default=100, help='Number of children for GA')
    parser.add_argument('--p_c', type=float, default=0.3, help='Crossover probability for GA')
    parser.add_argument('--p_m', type=float, default=0.3, help='Mutation probability for GA')
    parser.add_argument('--elite_size', type=int, default=10, help='Number of elite individuals for GA')
    parser.add_argument('--sigma', type=float, default=0.25, help='Initial standard deviation for CMA-ES')
    parser.add_argument('--tournament_size', type=int, default=20, help='Tournament size for GA')
    parser.add_argument('--seed', type=int, default=3, help='Random seed for reproducibility')
    parser.add_argument('--cost_conn', type=int, default=1, help='Connection criterion cost')
    parser.add_argument('--cost_area', type=int, default=1, help='Area criterion cost')
    parser.add_argument('--json_file', type=str, default=None, help='Specify the JSON file to load R, E, G_list, and a')
    parser.add_argument('--skip_cma', action='store_true', help='Flag to skip CMA-ES optimization step')
    parser.add_argument('--skip_local_search', action='store_true', help='Flag to skip local search steps')
    parser.add_argument('--skip_lp', action='store_true', help='Flag to skip Gurobi LP step')
    parser.add_argument('--plot', action='store_true', help='Plot evolution and placement of active pipeline stages')
    parser.add_argument('--save_performance', action='store_true', help='Save performance data to csv file')
    parser.add_argument('--profile', action='store_true', help='Print profiling data of active pipeline stages')
    
    return parser.parse_args()


    
if __name__ == "__main__":
    
    args = parse_arguments()

    N = args.N
    population_ga = args.population_ga
    population_cma = args.population_cma
    generations_ga = args.generations_ga
    generations_cma = args.generations_cma
    childs = args.childs
    p_c = args.p_c
    p_m = args.p_m
    elite_size = args.elite_size
    sigma = args.sigma
    seed = args.seed  # Read seed argument
    skip_cma = args.skip_cma
    skip_local_search = args.skip_local_search
    skip_lp = args.skip_lp
    plot = args.plot
    save_performance = args.save_performance
    profile = args.profile
    json_file = args.json_file  # Retrieve the json file argument
    cost_conn = args.cost_conn
    cost_area = args.cost_area
    tournament_size = args.tournament_size

    print(f"Area Cost               {cost_area:>5}")
    print(f"Connection Cost         {cost_conn:>5}")
    print(f"Population Size GA      {population_ga:>5}")
    print(f"Population Size CMA     {population_cma:>5}")
    print(f"Generations GA          {generations_ga:>5}")
    print(f"Generations CMA         {generations_cma:>5}")
    print(f"Crossover Probability   {p_c:>5.2f}")
    print(f"Mutation Probability    {p_m:>5.2f}")
    print(f"Elite Size              {elite_size:>5}")
    print(f"Tournament Size         {tournament_size:>5}")
    print(f"Sigma for CMA-ES        {sigma:>5.2f}")
    print(f"Random Seed             {seed:>5}")
    print(f"Skip CMA-ES             {skip_cma:>5}")
    print(f"Skip Local Search       {skip_local_search:>5}")
    print(f"Skip Gurobi LP          {skip_lp:>5}")
    print(f"Plot                    {plot:>5}")
    print(f"Save Performance        {save_performance:>5}")
    print(f"Profiling               {profile:>5}")
    print("\n")
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    W_max = 1000
    H_max = 1000

    # Load data from the specified JSON file if provided
    if json_file:
        R, G_list, E, a = deu.load_data_json(json_file)
    else:
        # If no JSON file is provided, initialize random data
        R, G_list, a = deu.init_rectangles_random(N, W_max, H_max, seed)
        E = deu.init_nets_random( N, int(N / 3), seed)
        if save_performance:
            deu.save_data_json(f"data/{dt.datetime.now()}.json", R, G_list, E, a)

    for i in range(N):
        for j in range(N):
            a[i][j] = a[j][i]
            
#---------------------------------------------------------------------------------------------------
#    Genetic algorithm
#---------------------------------------------------------------------------------------------------
    
    if profile:
        with cProfile.Profile() as profile:
            placed_ga, pm_ga, fitness_ga, ga_cpu_time, fitness_over_time_ga, chromosomes_over_time_ga, solution_ga = ga.run_ga(
                R, G_list, E, a, N, childs, p_c, p_m, elite_size, tournament_size, generations_ga, population_ga, 3 * N + 1, W_max, H_max, cost_conn, cost_area, True
            )
        # Print profiling for GA
        pstats.Stats(profile).sort_stats('cumulative').print_stats()
    else:
    # GA execution
        placed_ga, pm_ga, fitness_ga, ga_cpu_time, fitness_over_time_ga, chromosomes_over_time_ga, solution_ga = ga.run_ga(
            R, G_list, E, a, N, childs, p_c, p_m, elite_size, tournament_size, generations_ga, population_ga, 3 * N + 1, W_max, H_max, cost_conn, cost_area, False
        )

        # GA execution with profiling
        if save_performance:
            profiling_data = [
                ['N', 'Childs', 'Tournament size', 'Crossover prob', 'Mutation prob',  'Elite size', 'Generations', 'Population size', 'GA CPU time', 'Fitness','Date'],
                [N, childs, tournament_size, p_c, p_m, elite_size, generations_ga, population_ga, ga_cpu_time, fitness_ga, dt.datetime.now()]
            ]
            iou.save_performance_data('data/GA_profiling_data.csv', profiling_data, 
                                    ['N', 'Childs', 'Tournament size', 'Crossover prob', 'Mutation prob',  'Elite size', 'Generations', 'Population size']
                                   )

    
    # Plot evolution and placement for GA
    if  plot:
        iou.plot_evolution_pca(chromosomes_over_time_ga, fitness_over_time_ga, generations_ga, 'images/evolution_ga.png')
        iou.plot_placement(G_list, placed_ga, N, W_max, H_max, 'images/ga_placement.png')

#---------------------------------------------------------------------------------------------------
#    CMA-ES
#---------------------------------------------------------------------------------------------------
    
    # CMA-ES execution (if not skipped)
    if not skip_cma:
        initial_mean = solution_ga
        if profile:
            with cProfile.Profile() as profile:
                placed_cma, pm_cma, fitness_cma, cma_cpu_time, fitness_over_time_cma, chromosomes_over_time_cma = cov.run_cma(
                    R, G_list, E, a, N, W_max, H_max, population_cma, 3 * N + 1, initial_mean, generations_cma, sigma, cost_conn, cost_area, True
                )

            pstats.Stats(profile).sort_stats('cumulative').print_stats()
        else:
            placed_cma, pm_cma, fitness_cma, cma_cpu_time, fitness_over_time_cma, chromosomes_over_time_cma = cov.run_cma(
                R, G_list, E, a, N, W_max, H_max, population_cma, 3 * N + 1, initial_mean, generations_cma, sigma, cost_conn, cost_area, False
            )
            if save_performance:
                profiling_data = [
                    ['N', 'Generations', 'Sigma', 'Population size', 'CMA CPU time','Fitness','Date'],
                    [N, generations_cma, sigma, population_cma, cma_cpu_time, fitness_cma, dt.datetime.now()]
                ]
                iou.save_performance_data('data/CMA_profiling_data.csv', profiling_data, ['N', 'Generations', 'Sigma', 'Population size'])

        if plot:
            iou.plot_evolution_pca(chromosomes_over_time_cma, fitness_over_time_cma, generations_cma, 'images/evolution_cma.png')
            iou.plot_placement(G_list, placed_cma, N, W_max, H_max, 'images/cma_placement.png')

        if fitness_cma < fitness_ga:
            placed_meta = placed_cma
            fitness_meta = fitness_cma
            pm_meta = pm_cma
        else:
            placed_meta = placed_ga
            fitness_meta = fitness_ga
            pm_meta = pm_ga
    else:
        placed_meta = placed_ga
        fitness_meta = fitness_ga
        pm_meta = pm_ga

#---------------------------------------------------------------------------------------------------
#    Local searches
#---------------------------------------------------------------------------------------------------
    
    # Run Local Search if not skipped
    if not skip_local_search:
        if profile:
            with cProfile.Profile() as profile:

                placed_ls_seq, fitness_ls_seq, rectangles_ls_seq = ls.local_search_sequence(
                    R, G_list, E, a, N, placed_meta,fitness_meta, pm_meta, W_max, H_max,cost_conn, cost_area
                )
                

            pstats.Stats(profile).sort_stats('cumulative').print_stats()
            with cProfile.Profile() as profile:
                placed_ls_lay, fitness_ls_lay = ls.local_search_layout(
                    R, G_list, E, a, N, placed_ls_seq, rectangles_ls_seq, fitness_ls_seq, pm_meta, W_max, H_max, cost_conn, cost_area
                )

            pstats.Stats(profile).sort_stats('cumulative').print_stats()
        
        else:
            # Local search - Sequence
            start_time = time.process_time()
            placed_ls_seq, fitness_ls_seq, rectangles_ls_seq = ls.local_search_sequence(
                R, G_list, E, a, N, placed_meta, fitness_meta, pm_meta, W_max, H_max, cost_conn, cost_area
            )
            ls_seq_cpu_time = time.process_time() - start_time
    
    
            # Local search - Layout
            start_time = time.process_time()
            placed_ls_lay, fitness_ls_lay = ls.local_search_layout(
                R, G_list, E, a, N, placed_ls_seq, rectangles_ls_seq, fitness_ls_seq, pm_meta, W_max, H_max, cost_conn, cost_area
            )
            ls_lay_cpu_time = time.process_time() - start_time
            if save_performance:
                
                profiling_data = [
                    ['N', 'LS sequence CPU time', 'Date'],
                    [N, ls_seq_cpu_time, dt.datetime.now()]
                ]
                iou.save_performance_data('data/Localsearch_sequence_profiling_data.csv', profiling_data, ['N'])
                
                profiling_data = [
                    ['N', 'LS layout CPU time', 'Date'],
                    [N, ls_lay_cpu_time, dt.datetime.now()]
                ]
                iou.save_performance_data('data/Localsearch_layout_profiling_data.csv', profiling_data, ['N'])

        if plot:
            iou.plot_placement(G_list, placed_ls_seq, N, W_max, H_max, 'images/ls_seq_placement.png')
            iou.plot_placement(G_list, placed_ls_lay, N, W_max, H_max, 'images/ls_lay_placement.png')

        # Set the best placement after local search
        final_placement = placed_ls_lay
    else:
        final_placement = placed_meta

#---------------------------------------------------------------------------------------------------
#    Gurobi
#---------------------------------------------------------------------------------------------------
    
    if not skip_lp:
        if profile:
            with cProfile.Profile() as profile:
                lp_placed = lp.AMS_placement_gurobi(R, G_list, E, a, N, final_placement, cost_conn, cost_area)
            # Print profiling for Local Search Layout
            pstats.Stats(profile).sort_stats('cumulative').print_stats()
        else:
            start_time = time.process_time()
            lp_placed = lp.AMS_placement_gurobi(R, G_list, E, a, N, final_placement, cost_conn, cost_area)
            lp_cpu_time = time.process_time() - start_time
            if save_performance:
                profiling_data = [['N', 'Gurobi CPU time', 'Date'], [N, lp_cpu_time, dt.datetime.now()]
                ]
                iou.save_performance_data('data/LP_profiling_data.csv', profiling_data, ['N'])
        
        if plot:
            iou.plot_placement(G_list, lp_placed, N, W_max, H_max, 'images/lp_placement.png')

