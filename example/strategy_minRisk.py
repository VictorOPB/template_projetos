import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random
from modules.get_retornos_sp import get_retornos_sp
from modules.initial_weights import get_uniform_noneg

def _sel_stocks(returns, size):
    """
    Select top 'size' stocks based on standard deviation of returns.

    Args:
        returns (DataFrame): DataFrame containing returns data.
        size (int): Number of stocks to select.

    Returns:
        DataFrame: Selected returns data.
    """
    if (size is None) or (size > len(returns.columns)):
        size = len(returns.columns)
    stocks_sel = returns.std().sort_values().head(size).index
    returns_sel = returns[stocks_sel]
    return returns_sel

def _calculate_risk_stat(weights, returns):
    """
    Calculate the risk statistic based on portfolio weights and returns' covariance.

    Args:
        weights (array-like): Portfolio weights.
        returns (DataFrame): DataFrame containing returns data.

    Returns:
        float: Calculated risk statistic.
    """
    return (weights @ returns.cov() @ weights * 252) ** 0.5

def strategy_minRisk(data, t, size=30, window_size=500):
    """
    Implement a minimum risk strategy.

    Args:
        data (dict): Data dictionary containing 'sp' and 'prices' DataFrames.
        t (int): The desired time.
        size (int, optional): Number of stocks to consider. Defaults to 30.
        window_size (int, optional): Size of the window for calculations. Defaults to 500.

    Returns:
        DataFrame: Optimal weights for the selected stocks.
    """
    returns = get_retornos_sp(data, t, window_size)
    returns_sel = _sel_stocks(returns, size)
    initial_weights = get_uniform_noneg(size)

    # Define optimization constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights must equal 1
    ]

    # Define bounds for weights (0 <= weight <= 1)
    bounds = tuple((0, 1) for _ in range(size))

    # Perform optimization
    result = minimize(
        _calculate_risk_stat,
        initial_weights,
        args=(returns_sel,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    opt_weights = pd.DataFrame({
        'date': [data['prices'].index[t]] * len(result.x),
        'ticker': returns_sel.columns,
        'weights': result.x,
    })

    return opt_weights
