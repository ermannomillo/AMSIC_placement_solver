import os
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split



def save_performance_data(csv_file, performance_data, primary_keys):
    """Save / update performance data of placement stage """
    
    # If the file exists, read it
    if os.path.exists(csv_file):
        with open(csv_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)
        
        # Extract header from the data
        header = data[0]

        # Find the indices of the primary key columns in the header
        primary_key_indices = [header.index(key) for key in primary_keys]

        # Find the row that matches the primary key values in the new data (performance_data)
        for i, row in enumerate(data[1:], start=1):  # Skip header
            if all(row[idx] == str(performance_data[1][header.index(key)]) for idx, key in zip(primary_key_indices, primary_keys)):
                # Overwrite the row if all primary key columns match
                data[i] = performance_data[1]
                break
        else:
            # Append the new data if no matching row is found
            data.append(performance_data[1])
    else:
        # If the file does not exist, create it with the header and profiling data
        data = performance_data

    # Write data back to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)



def plot_placement(G_list, placed, N, W_max, H_max, name):
    """Plot and save on image spatial configuration of placement"""
    
    # Plotting the rectangles
    fig2, ax2 = plt.subplots()
    ax2.set_xlim(0, W_max)
    ax2.set_ylim(0, H_max)
    ax2.set_aspect('equal')
    ax2.set_xlabel('[nm]')
    ax2.set_ylabel('[nm]')
        
	# Generate a list of colors
    cmap = plt.get_cmap('magma')  # You can use any colormap you prefer
    colors = [cmap(i / N) for i in range(N)]

    vec_G_global = []
    
    for k in range(len(G_list)):
    
        vec_G = []    
        for key in G_list[k]:
            i, j, flag = G_list[k][key]
            vec_G.append(i)
            if flag == 0:
                vec_G.append(j)
                
        vec_G_global += vec_G
        
    for i in range(N):
        rect = plt.Rectangle((placed[i][0], placed[i][1]), placed[i][2], placed[i][3],
                                 linewidth=2, edgecolor='none', facecolor=colors[random.randint(0, N-1)])
        if placed[i][4] in vec_G_global :
        	rect = plt.Rectangle((placed[i][0], placed[i][1]), placed[i][2], placed[i][3],
                                 linewidth=2, edgecolor='r', facecolor=colors[random.randint(0, N-1)])
        ax2.add_patch(rect)
	
    fig2.savefig(name)

def plot_evolution_pca(chromosomes_over_time, fitness_over_time, generations, name):
    """Plot evolution of EA via PCA dimensionality reduction tecnique"""
    
    flatten_chromosomes = [chromosome for generation in chromosomes_over_time for chromosome in generation]
    flatten_fitness = [fitness for fitness_gen in fitness_over_time for fitness in fitness_gen]

    # Filter out chromosomes where fitness == 100000
    filtered_chromosomes = [chromosome for chromosome, fitness in zip(flatten_chromosomes, flatten_fitness) if fitness != 100000]
    filtered_fitness = [fitness for fitness in flatten_fitness if fitness != 100000]

    # Apply PCA to the filtered chromosomes
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(filtered_chromosomes)

    # Create the 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], filtered_fitness, 
                    c=np.linspace(1, generations+1, len(filtered_chromosomes)), cmap='plasma')

    # Add colorbar and labels
    cbar = plt.colorbar(sc, label='Generation')
    cbar.set_ticks(range(1, generations+1))
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('Fitness')
    ax.set_title('3D PCA of Chromosome Data Over Generations')
    ax.view_init(elev=5, azim=130)
    
    # Save the figure
    fig.savefig(name)




def plot_3d(df, x_col, y_col1, y_col2, title, xlabel, ylabel1, ylabel2):
    """Plot 3D graph"""
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract values for the axes
    x = df[x_col].values
    y1 = df[y_col1].values
    y2 = df[y_col2].values

    # Scatter plot the data points
    scatter = ax.scatter(x, y1, y2, label=ylabel1, color='blue', s=50)

    # Label the axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1)
    ax.set_zlabel(ylabel2)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Use mplcursors to add interactive hovering feature
    #cursor = mplcursors.cursor(scatter, hover=True)

    # Define the hover action: display the full row of data when hovering
    #@cursor.connect("add")
    def on_add(sel):
        # Get the index of the hovered point
        index = sel.target.index
        # Format hover text with the full row information
        sel.annotation.set_text(f"{df.iloc[index].to_dict()}")

    # Show the plot
    plt.show()

    

def plot_3d_with_colorbar(df, x_col, y_col1, y_col2, color_col, title, xlabel, ylabel1, ylabel2):
    """Plot 3D graph with an extra colorbar"""
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract values for the axes and the color column
    x = df[x_col].values
    y1 = df[y_col1].values
    y2 = df[y_col2].values
    colors = df[color_col].values  # Values to use for coloring the points

    # Scatter plot the data points with color mapping based on 'color_col'
    scatter = ax.scatter(x, y1, y2, c=colors, cmap='viridis', s=50)

    # Add color bar to show the mapping of color values
    color_bar = fig.colorbar(scatter, ax=ax, pad=0.1)
    color_bar.set_label(color_col)

    # Label the axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1)
    ax.set_zlabel(ylabel2)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()



def poly_fit(df, label, image_path, max_degree):
    """Polynomially fit data respect N and label"""
    
    df = df.sort_values(by='N').reset_index(drop=True)
        
    # Split the data into training and validation sets (80% train, 20% validate)
    N_train, N_val, y_train, y_val = train_test_split(df['N'], df[label], test_size=0.2, random_state=42)
        
    # Define the range of polynomial degrees to test
    degrees = range(1, max_degree+1)  # Limit the degrees to reduce the chance of overfitting
        
    # Initialize variables to store the best degree and its corresponding scores
    best_degree = None
    best_val_r2 = -np.inf  # Start with the lowest possible R-squared
        
    # Iterate over each degree and compute the fit
    for degree in degrees:
        # Fit the polynomial on the training data
        poly_coeffs = np.polyfit(N_train, y_train, degree)
            
        # Evaluate on the validation set
        y_val_pred = np.polyval(poly_coeffs, N_val)
            
        # Calculate R-squared and MSE for the validation set
        val_r2 = r2_score(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
            
        # Print the degree, R-squared, and MSE
        print(f'Degree: {degree}, Validation R-squared: {val_r2:.4f}, Validation MSE: {val_mse:.4f}')
            
        # Update the best degree if this one is better
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_degree = degree
            best_poly_coeffs = poly_coeffs

    # Generate the best fit polynomial on the entire dataset
    best_poly_vals = np.polyval(best_poly_coeffs, df['N'])
        
    # Print the best polynomial equation
    print("\nBest Polynomial Equation:")
    poly_eq = " + ".join([f"{coef:.4f}*x^{i}" for i, coef in enumerate(reversed(best_poly_coeffs))])
    print(f"y = {poly_eq}")
            
    # Plot the original data points and the best polynomial fit
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(df['N'], df[label], 'o', label=label)
    plt.plot(df['N'], best_poly_vals, '-', label=f'Best Polynomial Fit (degree {best_degree})')
        
    # Add labels and title
    plt.xlabel('N')
    plt.ylabel('CPU Time (s)')
    plt.title(f'Best Fit Polynomial (degree {best_degree})\nValidation R-squared: {best_val_r2:.4f}')
    plt.legend()
    plt.grid(True)
        
    fig2.savefig(image_path)

