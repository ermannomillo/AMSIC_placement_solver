import subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import argparse
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import IO_util as iou
import device_util as deu
import stats_util as stu


def run_command(command):
    """Task wrapper to run command in parallel."""
    print( f"Running: {command}" )
    subprocess.run(command, shell=True)
    

def has_been_executed_cma(df, N, population_size, sigma, generation):
    """Check if command has already been executed."""
    try:
        # Filter the DataFrame based on the provided parameters
        flag = df[
            (df['N'] == N) & 
            (df['Population size'] == population_size) & 
            (df['Sigma'] == sigma) & 
            (df['Generations'] == generation)
        ].empty
        # Return True if the command has been executed (i.e., the DataFrame is not empty)
        return not flag
    except Exception as e:
        return False

def has_been_executed_ga(df, N, population_size, elite_size, generation, childs, p_c, p_m, T):
    """Check if command has already been executed."""
    try:
        # Filter the DataFrame based on the provided parameters
        flag = df[
            (df['N'] == N) & 
            (df['Population size'] == population_size) & 
            (df['Elite size'] == elite_size) & 
            (df['Generations'] == generation) & 
            (df['Crossover prob'] == p_c) & 
            (df['Mutation prob'] == p_m) & 
            (df['Tournament size'] == T) &
            (df['Childs'] == ga_childs)
        ].empty
        # Return True if the command has been executed (i.e., the DataFrame is not empty)
        return not flag
    except Exception as e:
        return False

def load_data():
    """Load data from CSV files with error handling."""
    try:
        cma_df = pd.read_csv('data/CMA_profiling_data.csv')
        ga_df = pd.read_csv('data/GA_profiling_data.csv')
        return cma_df, ga_df
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None
        

def parse_arguments():
    """Parse console arguments"""    
    
    parser = argparse.ArgumentParser(description="Update and plot profiling data")
    parser.add_argument('--update', action='store_true', help='Update the profiling data by executing all commands')
    parser.add_argument('--profile', action='store_true', help='Run profiling using cProfile')
    parser.add_argument('--generate_benchmark', action='store_true', help='Generate placement contexts as benchmarks.')
    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_arguments()

    ga_population_sizes = [50, 100, 150, 200]
    ga_tournament_sizes = [5, 10, 20, 30]
    ga_elite_sizes = [5, 10, 20, 30]
    ga_childs = [5, 10, 20, 30]

    cma_population_sizes = [50, 100, 150, 200]
    cma_sigmas = [0.1, 0.25, 0.5, 1.0]

    # Generate hyperparameter combinations
    ga_hyperparams = list(product(ga_population_sizes, ga_elite_sizes, ga_tournament_sizes, ga_childs))
    cma_hyperparams = list(product(cma_population_sizes, cma_sigmas))

    # Instance sizes to test
    instance_sizes = [50]

    
    
    if args.generate_benchmark:
        for k in instance_sizes:
            R, G_list, a = deu.init_rectangles_random(k, 1000, 1000, 3)
            E = deu.init_nets_random( k, int(k / 3), 3)
            deu.save_data_json(f"tmp/context_N{k}.json", R, G_list, E, a)
    
    if args.update:

        commands = []
        for N in instance_sizes:
            cma_df, ga_df = load_data()
            for i in range(len(ga_hyperparams)):
                ga_population_size, ga_elite_size, ga_tournament, ga_child  = ga_hyperparams[i]
    
                # Only run CMA-ES when conditions are met and it hasn't been executed
                if ga_population_size == ga_population_sizes[0] and ga_elite_size == ga_elite_sizes[0]  and ga_tournament == ga_tournament_sizes[0] and ga_child == ga_childs[0]:
                    for j in range(len(cma_hyperparams)):
                        cma_population_size, cma_sigma = cma_hyperparams[j]
                        if not has_been_executed_cma(cma_df, N, cma_population_size, cma_sigma, 20):
                            command = (
                                f"python3 test_parallel.py {N} "
                                f"--tournament_size {ga_tournament} "
                                f"--generations_ga {20} "
                                f"--p_c {0.3} "
                                f"--p_m {0.3} "
                                f"--elite_size {ga_elite_size} "
                                f"--sigma {cma_sigma} "
                                f"--childs {ga_child} "
                                f"--generations_cma {20} "
                                f"--population_cma {cma_population_size} "
                                f"--population_ga {ga_population_size} "
                                f"--cost_conn=8 "
                                f"--skip_local_search "
                                f"--skip_lp "
                                f"--save_performance "
                                f"--json_file=tmp/context_N{N}.json"
                            )
                            commands.append(command)
                else:
                    # Run only GA optimization with --skip_cma
                    if not has_been_executed_ga(ga_df, N, ga_population_size, ga_elite_size, 20, ga_child, 0.3, 0.3, ga_tournament ):
                        command = (
                            f"python3 test_parallel.py {N} "
                            f"--population_ga {ga_population_size} "
                            f"--generations_ga {20} "
                            f"--tournament_size {ga_tournament} "
                            f"--p_c {0.3} "
                            f"--p_m {0.3} "
                            f"--childs {ga_child} "
                            f"--elite_size {ga_elite_size} "
                            f"--cost_conn=8 "
                            f"--skip_cma "
                            f"--skip_local_search "
                            f"--skip_lp "
                            f"--save_performance "
                            f"--json_file=tmp/context_N{N}.json"
                        )
                        commands.append(command)
        # Run all commands if the update flag is provided
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.map(run_command, commands)
    if args.profile:
        command = (f"python3 test_parallel.py {40} "
            f"--profile " 
            f"--json_file=tmp/context_N{40}.json")

    # CMA -------------------------------------------------------------------------------------
    
    df = pd.read_csv('data/CMA_profiling_data.csv')

    #iou.poly_fit(df[(df['Sigma'] == 0.1) & (df['Population size'] == 50)], 'CMA CPU time', 'images/GA_fit.png', 5)
    
    iou.plot_3d(df, 'N', 'CMA CPU time', 'Fitness',
                title='3D Plot: CMA CPU Time, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='CMA CPU Time', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Population size', 'Fitness',
                title='3D Plot: Population size, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Sigma', 'Fitness',
                title='3D Plot: Sigma, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Sigma', ylabel2='Fitness')
      
    iou.plot_3d_with_colorbar(df, 'N', 'Sigma', 'CMA CPU time', 'Population size',
            title='3D Plot: CMA CPU Time, Sigma, Population size vs N with Color', 
            xlabel='N (Problem Size)', ylabel1='Sigma', ylabel2='CMA CPU Time')
   
    
    independent_vars = ['Population size', 'Sigma', 'N']
    dependent_var_1 = 'CMA CPU time'
    dependent_var_2 = 'Fitness'
    
    # Running the function for GA CPU time
    print("\nDiagnostics for CMA CPU Time")
    stu.run_gls_with_diagnostics(df, dependent_var_1, independent_vars, use_glsar=True, standardize=True, use_newey_west=True)
    
    # Running the function for Fitness
    print("\nDiagnostics for CMA Fitness")
    stu.run_gls_with_diagnostics(df, dependent_var_2, independent_vars, use_glsar=True, standardize=True, use_newey_west=True)

    print("\n")

    # GA --------------------------------------------------------------------------------------
    df = pd.read_csv('data/GA_profiling_data.csv')
    
    iou.plot_3d(df, 'N', 'GA CPU time', 'Fitness',
                title='3D Plot: GA CPU Time, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='GA CPU Time', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Population size', 'Fitness',
                title='3D Plot: Population size, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Elite size', 'Fitness',
                title='3D Plot: Elite size, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Elite size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Childs', 'Fitness',
                title='3D Plot: Childs, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Childs', ylabel2='Fitness')

    iou.plot_3d(df, 'N', 'Tournament size', 'Fitness',
                title='3D Plot: Tournament size, Fitness vs N', 
                xlabel='N (Problem Size)', ylabel1='Tournament size', ylabel2='Fitness')

    iou.plot_3d(df, 'N', 'Population size', 'GA CPU time',
                title='3D Plot: Population size, GA CPU time vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='GA CPU time')
        
    iou.plot_3d(df, 'N', 'Elite size', 'GA CPU time',
                title='3D Plot: Elite size, GA CPU time vs N', 
                xlabel='N (Problem Size)', ylabel1='Elite size', ylabel2='GA CPU time')
        
    iou.plot_3d(df, 'N', 'Childs', 'GA CPU time',
                title='3D Plot: Childs, GA CPU time vs N', 
                xlabel='N (Problem Size)', ylabel1='Childs', ylabel2='GA CPU time')

    iou.plot_3d(df, 'N', 'Tournament size', 'GA CPU time',
                title='3D Plot: Tournament size, GA CPU time vs N', 
                xlabel='N (Problem Size)', ylabel1='Tournament size', ylabel2='GA CPU time')

    independent_vars = ['Population size', 'Elite size', 'N', 'Tournament size', 'Childs']
    dependent_var_1 = 'GA CPU time'
    dependent_var_2 = 'Fitness'
    
    # Running the function for GA CPU time
    print("\nDiagnostics for GA CPU Time")
    stu.run_gls_with_diagnostics(df, dependent_var_1, independent_vars, use_glsar=True, standardize=True, use_newey_west=True)
    
    # Running the function for Fitness
    print("\nDiagnostics for GA Fitness")
    stu.run_gls_with_diagnostics(df, dependent_var_2, independent_vars, use_glsar=True, standardize=True, use_newey_west=True)

    # LS & ILP ---------------------------------------------------------------------------------

    print("\n")
    
    df = pd.read_csv('data/Localsearch_sequence_profiling_data.csv')
    stu.poly_fit(df, 'LS sequence CPU time', 'images/LS_seq_fit.png', 7)
    
    df = pd.read_csv('data/Localsearch_layout_profiling_data.csv')
    stu.poly_fit(df, 'LS layout CPU time', 'images/LS_lay_fit.png', 7)

    df = pd.read_csv('data/LP_profiling_data.csv')
    stu.poly_fit(df, 'Gurobi CPU time', 'images/LP_fit.png', 7)
    

      
            