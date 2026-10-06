import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization

# Set random number generator with a seed of 42
rng = np.random.default_rng(42)

# Generate a random expected returns vector mu for 6 assets
mu = rng.random(6)

# Generate a 6x6 positive semi-definite covariance matrix sigma
# (by generating a random matrix A and computing A @ A.T)
A = rng.random((6, 6))
sigma = A @ A.T

# Create a PortfolioOptimization instance
portfolio = PortfolioOptimization(
    expected_returns=mu,
    covariances=sigma,
    risk_factor=0.5,
    budget=3
)

# Convert the portfolio optimization problem into a quadratic program
qp = portfolio.to_quadratic_program()

# Print qp to verify
print(qp)
