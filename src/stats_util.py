import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.linear_model import GLSAR
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import PolynomialFeatures, StandardScaler



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



def run_gls_with_diagnostics(df, dependent_var, independent_vars, use_glsar=False, standardize=False, use_newey_west=False):
    """
    Runs GLS regression with a log-transformed dependent variable, calculates diagnostics 
    such as VIF, Durbin-Watson statistic, and condition number, and prints the results.

    Parameters:
    df (DataFrame): The input dataframe containing the data.
    dependent_var (str): The name of the dependent variable.
    independent_vars (list): List of independent variables.
    use_glsar (bool): If True, use Cochrane-Orcutt method for autocorrelation.
    standardize (bool): If True, standardize the independent variables (mean=0, std=1).
    use_newey_west (bool): If True, adjust standard errors using Newey-West covariance.

    Prints:
    Model summary, VIF values, Durbin-Watson statistic, and Condition Number.
    """
    
    # Apply log transformation to the dependent variable
    log_dep_var = 'log_' + dependent_var
    df.loc[:, log_dep_var] = np.log(df[dependent_var])

    # Prepare independent variables (X) and log-transformed dependent variable (y)
    X = df[independent_vars]
    y = df[log_dep_var]

    # Standardize the independent variables if requested
    if standardize:
        scaler = StandardScaler()
        X.loc[:, independent_vars] = scaler.fit_transform(X[independent_vars])

    # Add a constant (intercept) to the independent variables
    X = add_constant(X)

    # GLSAR (Cochrane-Orcutt) to handle autocorrelation
    if use_glsar:
        model_log = GLSAR(y, X).iterative_fit(maxiter=10)
        print("\n--- Running GLSAR (Cochrane-Orcutt) ---")
    else:
        model_log = OLS(y, X).fit()

    # Print the model summary
    print("\nModel Summary:")
    if use_newey_west:
        # Use Newey-West standard errors
        cov_newey_west = cov_hac(model_log, nlags=1)
        print(model_log.get_robustcov_results(cov_type='HAC', use_t=True, maxlags=1).summary())
    else:
        print(model_log.summary())

    # Calculate VIF for each independent variable (excluding the constant)
    vif_data = pd.DataFrame()
    vif_data['Feature'] = independent_vars
    vif_data['VIF'] = [variance_inflation_factor(X.values, i + 1) for i in range(X.shape[1] - 1)]

    # Print VIF values
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)

    # Durbin-Watson statistic to detect autocorrelation
    dw_stat = durbin_watson(model_log.resid)
    print("\nDurbin-Watson Statistic:", dw_stat)

    # Condition Number to detect multicollinearity
    cond_number = np.linalg.cond(X)
    print("Condition Number:", cond_number)

    # Suggest further steps based on diagnostics
    if dw_stat < 1.5 or dw_stat > 2.5:
        print("\nWarning: Durbin-Watson statistic indicates potential autocorrelation. Consider using GLSAR or Newey-West standard errors.")
    
    if cond_number > 30:
        print("\nWarning: High condition number indicates multicollinearity. Consider removing or standardizing variables, or using Ridge regression.")