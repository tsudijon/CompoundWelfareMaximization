#import cvxpy as cp
import numpy as np
import scipy
from scipy.stats import poisson
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from scipy.integrate import cumtrapz
from scipy.special import sici  # returns (Si(x), Ci(x))

import matplotlib.pyplot as plt

def coupled_bootstrap(Xs, sigmas,eps = 1, lb = -5, ub = 5):
    """
    eps: tuning parameter for the coupled bootstrap risk
    """
    n = len(Xs)
    eps = 1./n**0.2

    C_grid = np.linspace(lb, ub, 100)

    welfare_curve = np.zeros_like(C_grid).astype(float)
    for i in np.arange(len(C_grid)):
        c = C_grid[i]
        F1 = Xs*norm.cdf((Xs - c*sigmas)/(eps*sigmas))
        F2 = sigmas*norm.pdf((Xs - c*sigmas)/(eps*sigmas))/eps
        welfare_curve[i] = F1.sum() - F2.sum()
    return welfare_curve

#### Near-Unbiased Welfare Estimates 

def dirichlet_normal_estimator(
    Xs, sigmas, lb=-5, ub=5, true_effects=None, plot=False
):
    """
    Vectorized welfare estimator using sine integral SI and unnormalized sinc(x) = sin(x)/(pi*x).

    Parameters
    ----------
    Xs : array-like, shape (n,)
        Observed estimates X_i.
    sigmas : array-like, shape (n,)
        Std devs sigma_i corresponding to X_i.
    lb, ub : float
        Grid bounds for decision threshold C.
    true_effects : array-like, optional
        If provided, plot the 'true' welfare curve used in your original template.
    smoothen_nn : bool
        If True, calls smoothen_via_nn(welfare_curve, sigmas, n, C_grid).
    plot : bool
        If True, plots the components and welfare curve.

    Returns
    -------
    welfare_curve : ndarray, shape (m,)
        Welfare evaluated on the C_grid.
    """
    Xs = np.asarray(Xs, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    ts = Xs/sigmas
    n = len(Xs)
    assert sigmas.shape == Xs.shape, "Xs and sigmas must have the same shape."

    C_grid = np.linspace(lb, ub, 100)
    m = C_grid.size

    h_n = 1./np.sqrt(2 * np.log(n))
    u = (ts[:, None] - C_grid[None, :]) / h_n

    # SI(u): use scipy.special.sici, first element is Si
    Si_u, _ = sici(u)
    sinc_u = np.sinc(u / np.pi) / np.pi

    T1 = 0.5 * np.sum(Xs) * np.ones(m)
    T2 = (Xs[:, None] * Si_u).sum(axis=0) / np.pi
    T3 = -((sigmas[:, None] / h_n) * sinc_u).sum(axis=0)

    welfare_curve = T1 + T2 + T3

    if true_effects is not None:
        true_effects = np.asarray(true_effects, dtype=float)
        assert true_effects.shape == Xs.shape, "true_effects must match Xs shape."
        true_risk = np.zeros_like(C_grid)
        te_over_sigma = true_effects / sigmas
        for j in range(m):
            true_risk[j] = np.sum(true_effects * norm.cdf(te_over_sigma - C_grid[j]))
        if plot:
            print("True Optimal Threshold:", C_grid[np.argmax(true_risk)])
            plt.plot(C_grid, true_risk, color="red", linestyle="dashed", alpha=0.6, label="True Welfare")
    if plot:
        plt.legend()
    return welfare_curve

def heteroskedastic_normal_welfare_estimator(Xs, sigmas,lb = -5, ub = 5, true_effects = None, smoothen_nn = False, plot = False):
    n = len(Xs)
    ts = Xs/sigmas

    C_grid = np.linspace(lb, ub, 100)

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
        # Ensure x is a 1D tensor of shape (N,) 
        w_expanded = self.w.unsqueeze(0)  
        x_expanded = x.unsqueeze(1)       

        result = w_expanded * self.gaussian_cdf(w_expanded/self.sigmas - x_expanded)

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


  return y_pred



#############################
##### Visualization #########
#############################

def compare_insample_welfares(welfares, labels, title):
    """
    welfares: list of arrays
    labels: list of strings
    """

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplots
    box_data = welfares

    # Box plot with customization
    boxplot = ax.boxplot(box_data, patch_artist=True, labels=labels)

    # Generate colors based on the number of boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
    for box, color in zip(boxplot['boxes'], colors):
        box.set(facecolor=color, alpha=0.7)
        
    # Add a horizontal grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add labels and title
    ax.set_title('Algorithm Performance Comparison', fontsize=16, pad=20)
    ax.set_ylabel('Welfare', fontsize=14)
    ax.set_xlabel('Algorithms', fontsize=14)

    # Add some padding to y-axis for better visualization
    y_min = min([min(data) for data in box_data])
    y_max = max([max(data) for data in box_data])
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)

    # Optional: Add data points
    for i, data in enumerate(box_data):
        # Add a small horizontal jitter to better visualize the points
        x = np.random.normal(i+1, 0.02, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=5, color='black')

    # Add some statistics in the plot
    for i, data in enumerate(box_data):
        stats_text = f"Mean: {np.mean(data):.2f}\nStd: {np.std(data):.2f}"
        ax.annotate(stats_text, xy=(i+1, y_max), xytext=(0, -10),
                    textcoords='offset points', ha='center', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.1))

    plt.tight_layout()
    #plt.savefig(title, bbox_inches = 'tight')
    plt.show()

    return None
