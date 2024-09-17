import subprocess
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import product
import argparse
from numpy.polynomial import Polynomial
import sys
import os

src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)

import IO_util as iou
import device_util as deu



def run_command(command):
    """Task wrapper to run command in parallel."""
    
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

def has_been_executed(df, N, population_size, sigma=None, elite_size=None):
    """Check if command has already been executed """
    
    if sigma is not None:
        return not df[(df['N'] == N) & (df['Population size'] == population_size) & (df['Sigma'] == sigma)].empty
    else:
        return not df[(df['N'] == N) & (df['Population size'] == population_size) & (df['Elite size'] == elite_size)].empty

def parse_arguments():
    """Parse console arguments"""    
    
    parser = argparse.ArgumentParser(description="Update and plot profiling data")
    parser.add_argument('--update', action='store_true', help='Update the profiling data by executing all commands')
    parser.add_argument('--profile', action='store_true', help='Run profiling using cProfile')
    return parser.parse_args()



if __name__ == "__main__":
    
    args = parse_arguments()

    ga_population_sizes = [50, 100, 150, 200]
    ga_elite_sizes = [5, 10, 20, 30]

    cma_population_sizes = [50, 100, 150, 200]
    cma_sigmas = [0.1, 0.25, 0.5, 1.0]

    # Generate hyperparameter combinations
    ga_hyperparams = list(product(ga_population_sizes, ga_elite_sizes))
    cma_hyperparams = list(product(cma_population_sizes, cma_sigmas))

    # Instance sizes to test
    instance_sizes = [20, 40, 60, 80]

    cma_df = pd.read_csv('data/CMA_profiling_data.csv')
    ga_df = pd.read_csv('data/GA_profiling_data.csv')

    if args.update:
        '''
        for k in instance_sizes:
            I = range(k)
            R, G_list, a = deu.init_rectangles_random(k, 1000, 1000)
            E = deu.init_nets_random(I, k, int(k / 3))
            deu.save_data_json(f"tmp/context_N{k}.json", R, G_list, E, a)
        '''
        commands = []
        for N in instance_sizes:
            for i in range(len(ga_hyperparams)):
                ga_population_size, ga_elite_size = ga_hyperparams[i]
    
                # Only run CMA-ES when conditions are met and it hasn't been executed
                if ga_population_size == ga_population_sizes[0] and ga_elite_size == ga_elite_sizes[0]:
                    for j in range(len(cma_hyperparams)):
                        cma_population_size, cma_sigma = cma_hyperparams[j]
                        if not has_been_executed(cma_df, N, cma_population_size, sigma=cma_sigma):
                            command = (
                                f"python3 test_parallel.py {N} "
                                f"--population_size {ga_population_size} "
                                f"--generations_ga {40} "
                                f"--p_c {0.3} "
                                f"--p_m {0.3} "
                                f"--elite_size {ga_elite_size} "
                                f"--sigma {cma_sigma} "
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
                    if not has_been_executed(ga_df, N, ga_population_size, elite_size=ga_elite_size):
                        command = (
                            f"python3 test_parallel.py {N} "
                            f"--population_ga {ga_population_size} "
                            f"--generations_ga {40} "
                            f"--p_c {0.3} "
                            f"--p_m {0.3} "
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

    df = pd.read_csv('data/CMA_profiling_data.csv')

    iou.poly_fit(df[(df['Sigma'] == 0.1) & (df['Population size'] == 50)], 'CMA CPU time', 'images/GA_fit.png', 5)
    
    iou.plot_3d(df, 'N', 'CMA CPU time', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='CMA CPU Time', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Population size', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Sigma', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Sigma', ylabel2='Fitness')
      
    iou.plot_3d_with_colorbar(df, 'N', 'Sigma', 'CMA CPU time', 'Population size',
            title='3D Plot: CPU Time, Fitness, Generations vs N with Color', 
            xlabel='N (Problem Size)', ylabel1='Sigma', ylabel2='CMA CPU Time')

    df = pd.read_csv('data/GA_profiling_data.csv')
    
    iou.plot_3d(df, 'N', 'GA CPU time', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='GA CPU Time', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Population size', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Elite size', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Elite size', ylabel2='Fitness')
        
    iou.plot_3d(df, 'N', 'Childs', 'Fitness',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Childs', ylabel2='Fitness')

    iou.plot_3d(df, 'N', 'Population size', 'GA CPU time',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Population size', ylabel2='GA CPU time')
        
    iou.plot_3d(df, 'N', 'Elite size', 'GA CPU time',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Elite size', ylabel2='GA CPU time')
        
    iou.plot_3d(df, 'N', 'Childs', 'GA CPU time',
                title='3D Plot: CPU Time, Fitness, Generations vs N', 
                xlabel='N (Problem Size)', ylabel1='Childs', ylabel2='GA CPU time')

    df = pd.read_csv('data/Localsearch_sequence_profiling_data.csv')
    iou.poly_fit(df, 'LS sequence CPU time', 'images/LS_seq_fit.png', 5)
    
    df = pd.read_csv('data/Localsearch_layout_profiling_data.csv')
    iou.poly_fit(df, 'LS layout CPU time', 'images/LS_lay_fit.png', 5)

    df = pd.read_csv('data/LP_profiling_data.csv')
    iou.poly_fit(df, 'Gurobi CPU time', 'images/LP_fit.png', 5)
    

      
            