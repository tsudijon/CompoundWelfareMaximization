#import cvxpy as cp
import numpy as np
import scipy
from scipy.stats import poisson
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

import matplotlib.pyplot as plt

def npmle_poisson(data):

  m = 500
  theta_grid = np.linspace(0.1, 50, m)
  g = cp.Variable(m, nonneg=True)  # Probability mass function on the grid

  # Constraints: g is a probability distribution
  constraints = [cp.sum(g) == 1]

  # Log-likelihood for the Poisson observations
  W = poisson.pmf(data.reshape(-1,1), theta_grid)
  log_likelihood = cp.sum(cp.log(W @ g))

  objective = cp.Maximize(log_likelihood)
  problem = cp.Problem(objective, constraints)
  problem.solve(verbose = True)

  return theta_grid, g

def compute_post_expectation(y, theta_grid, g):
  probs = poisson.pmf(y,theta_grid) * g.value
  probs = probs/probs.sum()
  return (theta_grid*probs).sum()


def coupled_bootstrap(Xs, sigmas,eps = 1, lb = -5, ub = 5):
    """
    eps: tuning parameter for the coupled bootstrap risk
    """
    n = len(Xs)

    C_grid = np.linspace(lb, ub, 500)

    # # Compute pairwise differences
    # d = Xs[:, None] - C_grid[None, :]
    # d = d/(eps*sigmas[:,None])

    # F_1 = (Xs[:,None]*norm.cdf(d)).sum(axis = 0)
    # F_2 = ((sigmas[:,None]/eps)*norm.pdf(d)).sum(axis = 0)
    # welfare_curve = F_1 - F_2

    welfare_curve = np.zeros_like(C_grid).astype(float)
    for i in np.arange(len(C_grid)):
        c = C_grid[i]
        F1 = Xs*norm.cdf((Xs - c*sigmas)/(eps*sigmas))
        F2 = sigmas*norm.pdf((Xs - c*sigmas)/(eps*sigmas))/eps
        welfare_curve[i] = F1.sum() - F2.sum()
    return welfare_curve


#### Near-Unbiased Welfare Estimates 


def heteroskedastic_normal_welfare_estimator(Xs, sigmas,lb = -5, ub = 5, true_effects = None, smoothen_nn = False, plot = False):
    n = len(Xs)
    ts = Xs/sigmas

    C_grid = np.linspace(lb, ub, 500)

    # Compute F_1: Sum of Xs for all elements greater than each C_grid point
    F_1 = np.sum(Xs[:, None] * (ts[:, None] >= C_grid), axis=0)

    # Compute pairwise differences
    d = ts[:, None] - C_grid[None, :]
    h_n = np.sqrt(2*np.log(n))
    F_2 = (sigmas[:,None]*np.sin(h_n*d)/(np.pi*d)).sum(axis = 0) # another kernel estimate 
    welfare_curve = F_1 - F_2

    if plot:
        plt.figure(figsize = (10,6))
        plt.plot(C_grid, F_1, linewidth = 1, alpha = 0.05, linestyle = "dashed")
        plt.plot(C_grid, F_2, linewidth = 1, alpha = 0.05, linestyle = "dashed")
        plt.plot(C_grid, welfare_curve, label = "Welfare Estimate", linewidth = 1, alpha = 0.75, color = "orange")
        plt.xlabel("Decision Threshold $C$")
        plt.ylabel("Welfare")
        print("NUWE Optimal Threshold:", C_grid[np.argmax(welfare_curve)])

    if true_effects is not None:
        true_risk = np.zeros_like(C_grid)
        for i in range(len(C_grid)):
            true_risk[i] = np.sum(true_effects*norm.cdf(true_effects/sigmas - C_grid[i]))

        if plot:
            print("True Optimal Threshold:", C_grid[np.argmax(true_risk)])
            plt.plot(C_grid, true_risk, color = "red", linestyle = "dashed", alpha = 0.5, label = "True Welfare")

    if smoothen_nn:
        smoothed_welfare_curve = smoothen_via_nn(welfare_curve, sigmas, n, C_grid).detach().numpy()

        if plot:
            plt.plot(C_grid, smoothed_welfare_curve, color = "orange", linestyle = "dashed", alpha = 1, label = "Smoothened Estimate", linewidth = 3)

    if plot:
        plt.legend()

    return welfare_curve


#### smoothing via neural network


# Define the model with Gaussian CDF activation
class OneLayerNN(nn.Module):
    def __init__(self, num_units, sigmas):
        super().__init__()
        self.w = nn.Parameter(torch.randn(num_units))  # Shared weights and biases
        self.sigmas = torch.Tensor(sigmas) #sigmas

    def gaussian_cdf(self, z):
        """Computes the Gaussian CDF using PyTorch's normal distribution."""
        return dist.Normal(0, 1).cdf(z)

    def forward(self, x):
        # Ensure x is a 1D tensor of shape (N,) where N=2000.
        # Expand dimensions for broadcasting:
        w_expanded = self.w.unsqueeze(0)  # Shape: (1, 200)
        x_expanded = x.unsqueeze(1)         # Shape: (2000, 1)

        # Now w_expanded - x_expanded will have shape (2000, 200)
        result = w_expanded * self.gaussian_cdf(w_expanded/self.sigmas - x_expanded)

        # Sum over the hidden units dimension (the last dimension, 200)
        return torch.sum(result, dim=-1)


def smoothen_via_nn(ure, sigmas, n, grid):
  """
  grid: some grid of values in R
  ure: estimate of the welfare of function evaluated at each of the grid.

  ## Need to adapt this to the heteroskedastic case
  """

  # Hyperparameters
  num_units = n    # Number of hidden units
  lr = 0.01         # Learning rate
  epochs = 5000     # Number of training epochs

  # Create model and optimizer
  model = OneLayerNN(num_units, sigmas)
  optimizer = optim.Adam(model.parameters(), lr=lr)  # Change to SGD if needed
  loss_fn = nn.MSELoss()

  x_train = torch.tensor(grid, dtype=torch.float32)
  y_train = torch.tensor(ure, dtype=torch.float32)

  # Training loop
  for epoch in range(epochs):
      optimizer.zero_grad()
      y_pred = model(x_train)
      loss = loss_fn(y_pred, y_train.squeeze())
      loss.backward()
      optimizer.step()

      if epoch % 100 == 0:
          print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

  print("Training complete.")

  return y_pred
