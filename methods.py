# import cvxpy as cp
import numpy as np
from scipy.stats import norm
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.special import sici  # returns (Si(x), Ci(x))

import matplotlib.pyplot as plt
from npeb import GLMixture

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter
from rpy2.robjects import numpy2ri, pandas2ri



def CLOSE(Ys, sigmas):
    np_pd_conv = default_converter + numpy2ri.converter + pandas2ri.converter

    close_pkg = importr("close")

    with conversion.localconverter(np_pd_conv):
        
        y_R = ro.FloatVector(Ys.tolist())
        s_R = ro.FloatVector(sigmas.tolist())
        res = close_pkg.compute_close(y_R, s_R)

    return res

def npmle_gaussian_hetero_rebayes(Ys, sigmas, v=300):
    """
    Gaussian heteroskedastic NPMLE via REBayes::GLmix.
    Parameters
    ----------
    Ys : array-like, shape (n,)
        Observations.
    sigmas : array-like, shape (n,)
        Known noise SDs.
    v : int or vector
        Grid control passed to GLmix (default 300).
    use_predict : bool
        If True, return posterior means via predict(..., Loss=2) instead of fit$dy.
    Returns
    -------
    posterior_means : np.ndarray, shape (n,)
    fit : rpy2 R object (GLmix fit) with fields like x (grid), y (mixing masses), g, dy, logLik, status.
    """
    conv = default_converter + numpy2ri.converter

    REBayes = importr("REBayes")  
    
    if Ys.shape[0] != sigmas.shape[0]:
        raise ValueError(f"len(Ys)={Ys.size} must equal len(sigmas)={sigmas.size}")

    # Convert inputs to R vectors explicitly
    with conversion.localconverter(conv):
        y_R = ro.FloatVector(Ys.tolist())
        s_R = ro.FloatVector(sigmas.tolist())
        
    fit_R = REBayes.GLmix(y_R, sigma=s_R, v=v) 
    post_R = fit_R.rx2("dy")

    # Convert the posterior means to NumPy
    posterior_means = np.asarray(conversion.rpy2py(post_R), dtype=float)
    G_grid = np.asarray(conversion.rpy2py(fit_R.rx2("x")), dtype=float)
    G_values = np.asarray(conversion.rpy2py(fit_R.rx2("y")), dtype=float)
    return posterior_means, (G_grid, G_values)

def npeb_npmle(Ys, sigmas, atoms = None):
    """
    Ys: Gaussian observations, 1D numpy array
    sigmas: standard errors of the observations, 1D numpy array
    """
    Y = Ys.reshape(-1,1)
    prec = (1./sigmas**2).reshape(-1,1)
    atoms = atoms.reshape(-1,1)

    m = GLMixture(prec_type='diagonal', atoms_init = atoms)

    ## Compute the NPMLE 
    m.fit(Y, prec, verbose = True)

    posterior_means = m.posterior_mean(Y, prec) 
    return posterior_means[:,0]


def fay_heriott(Ys, sigmas, Xs, max_iter=80, tol=1e-8):
    """
    Fay-Herriot area-level EB shrinkage with A estimated by the FH moment equation (Eq. 2.21).

    Parameters
    ----------
    Ys : array-like, shape (m,)
        Direct (design-based) estimators y_i for each area.
    sigmas : array-like, shape (m,)
        Known standard errors for the direct estimators. (We treat D_i = sigmas_i**2.)
    Xs : array-like, shape (m, p)
        Area-level covariates/design matrix X.
    max_iter : int
        Max bisection iterations for solving Eq. (2.21).
    tol : float
        Absolute tolerance on the root (in the equation value).

    Returns
    -------
    out : dict with keys
        A_hat : float
            Estimated between-area variance A (truncated at 0 if no positive solution).
        beta_hat : ndarray, shape (p,)
            GLS estimate at A_hat.
        theta_hat : ndarray, shape (m,)
            EBLUPs of small-area means.
        y_star : ndarray, shape (m,)
            Fitted regression means X beta_hat.
        shrinkage_B : ndarray, shape (m,)
            B_i = D_i / (A_hat + D_i)  (amount of shrinkage toward regression mean).
        weights_sample : ndarray, shape (m,)
            A_hat / (A_hat + D_i)  (weight on Y_i).
        weights_reg : ndarray, shape (m,)
            D_i / (A_hat + D_i)  (weight on X_i' beta_hat).

    Notes
    -----
    Implements FH eqs. (2.19), (2.21), (2.22). If no positive root of (2.21) exists,
    follows Fay–Herriot and sets A_hat = 0.
    """
    y = np.asarray(Ys, dtype=float).reshape(-1)
    D = np.asarray(sigmas, dtype=float).reshape(-1)**2  # treat inputs as standard errors
    X = np.asarray(Xs, dtype=float)
    m, p = X.shape
    if y.shape[0] != m or D.shape[0] != m:
        raise ValueError("Shapes mismatch: len(Ys), len(sigmas), and Xs.shape[0] must match.")

    if m <= p:
        raise ValueError("Need m > p to solve the FH moment equation.")

    # helper: GLS beta for a given A
    def beta_hat_given_A(A):
        Vinv = 1.0 / (A + D)
        XtVinv = (X.T * Vinv)     # p x m
        XtVinvX = XtVinv @ X      # p x p
        XtVinvY = XtVinv @ y      # p
        # Solve XtVinvX * beta = XtVinvY
        beta = np.linalg.solve(XtVinvX, XtVinvY)
        return beta

    # f(A) = sum ((y - X beta(A))^2 / (A + D)) - (m - p)
    def fh_moment_equation(A):
        beta = beta_hat_given_A(A)
        resid = y - X @ beta
        return np.sum((resid**2) / (A + D)) - (m - p)

    # Root finding for A >= 0
    f0 = fh_moment_equation(0.0)
    if f0 < 0:
        A_hat = 0.0
    else:
        lo, hi = 0.0, None
        # expand an upper bracket until sign change or cap reached
        candidate = max(1.0, np.median(D) if np.all(np.isfinite(D)) else 1.0)
        for _ in range(60):
            val = fh_moment_equation(candidate)
            if val < 0:
                hi = candidate
                break
            candidate *= 2.0
        if hi is None:
            # monotone but no sign change found in range -> default to zero as per FH spirit
            A_hat = 0.0
        else:
            # bisection
            lo = 0.0
            for _ in range(max_iter):
                mid = 0.5 * (lo + hi)
                fmid = fh_moment_equation(mid)
                if abs(fmid) <= tol:
                    A_hat = mid
                    break
                if fmid > 0:
                    lo = mid
                else:
                    hi = mid
            else:
                A_hat = 0.5 * (lo + hi)

    # Final GLS, fitted means, and EBLUPs
    beta_hat = beta_hat_given_A(A_hat)
    y_star = X @ beta_hat
    w_sample = A_hat / (A_hat + D)
    w_reg = D / (A_hat + D)
    theta_hat = w_sample * y + w_reg * y_star
    B = w_reg  # classic FH shrinkage factor toward regression mean

    return {
        "A_hat": float(A_hat),
        "beta_hat": beta_hat,
        "theta_hat": theta_hat,
        "y_star": y_star,
        "shrinkage_B": B,
        "weights_sample": w_sample,
        "weights_reg": w_reg,
    }


############################################
############################################
#### True welfare curves #######
############################################
############################################


def true_welfare_pvalue_curve(mus, sigmas, costs, grid):
    """
    TODO: refactor this so it takes in a general class of decisions
    This is written for pvalue strategy: Y_i > K_i + beta*sigma_i
    """
    true_risk = np.zeros_like(grid)
    diff = mus - costs
    for i in range(len(grid)):
        true_risk[i] = np.sum(diff*norm.cdf(diff/sigmas - grid[i]))
    return true_risk


############################################
############################################
#### Near-Unbiased Welfare Estimates #######
############################################
############################################

def coupled_bootstrap(Ys, sigmas, threshold_fn, param_grid, costs = None, eps=None):
    Ys = np.asarray(Ys, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    n = Ys.shape[0]
    n_grid = len(param_grid)

    if eps is None:
        eps = n ** (-0.2)

    thresholds = threshold_fn(param_grid)
    assert thresholds.shape == (n, n_grid), "threshold_fn must return (n, n_grid) array."


    net_benefit = Ys - costs

    us = (Ys[:, None] - thresholds) / (eps * sigmas[:, None])
    F1 = net_benefit[:, None] * norm.cdf(us)
    F2 = sigmas[:, None] * norm.pdf(us) / eps

    welfares = F1 - F2
    welfare = welfares.mean(axis=0)
    welfare_se = np.sqrt(((welfares - welfare) ** 2).mean(axis=0) / n)
    return welfare, welfare_se




def assure(Ys, sigmas, threshold_fn, param_grid, costs=None):
    """
    Vectorized welfare estimator using sine integral Si and unnormalized sinc(x)
    for general threshold functions. Assumes decision rules are of the form
    1(X_i > threshold_fn(param_grid))

    """
    n = len(Ys)
    n_grid = len(param_grid)
    thresholds = threshold_fn(param_grid)
    assert thresholds.shape == (n, n_grid), "threshold_fn must be vectorized."

    lmbda_n = np.sqrt(2 * np.log(n))

    us = lmbda_n * (Ys[:, None] - thresholds) / sigmas[:, None]  # shape (n, n_grid)
    Si_u, _ = sici(us)
    sinc_u = np.sinc(us / np.pi) / np.pi

    net_benefit = Ys - costs
    welfares = (
        (net_benefit / 2)[:, None]
        + (net_benefit / np.pi)[:, None] * Si_u
        - (sigmas * lmbda_n)[:, None] * sinc_u
    )

    welfare = (welfares).mean(axis=0)
    welfare_se = np.sqrt(((welfares - welfare) ** 2).mean(axis=0) / n)

    return welfare, welfare_se



def FIE(Ys, sigmas, threshold_fn, param_grid, costs = None):
    Ys = np.asarray(Ys, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    n = Ys.shape[0]
    n_grid = len(param_grid)

    thresholds = threshold_fn(param_grid)
    assert thresholds.shape == (n, n_grid), "threshold_fn must return (n, n_grid) array."

    net_benefit = Ys - costs

    F1 = net_benefit[:, None] * (Ys[:, None] >= thresholds)

    lmbda_n = np.sqrt(2 * np.log(n))
    us = lmbda_n * (Ys[:, None] - thresholds) / sigmas[:, None]  # shape (n, n_grid)
    sinc_u = np.sinc(us / np.pi) / np.pi

    F2 = (sigmas * lmbda_n)[:, None] * sinc_u

    welfares = F1 - F2
    welfare = welfares.mean(axis=0)
    welfare_se = np.sqrt(((welfares - welfare) ** 2).mean(axis=0) / n)
    return welfare, welfare_se

##########################################
##########################################
##### Decision Threshold Classes #########
##########################################
##########################################


def linear_threshold_fn(params, *, sigmas, costs, X):
    """
    params: array with shape (*P, 2), last axis = (C, b)
    returns: array with shape (n, *P)
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 2:
        raise ValueError("params[..., 2] must hold (C, b).")

    C, b = np.moveaxis(params, -1, 0)   # each has shape (*P)

    # Shapes
    n = np.asarray(sigmas).shape[0]
    P = C.shape  # tuple of grid dims

    # Expand sample-wise arrays to (n, *P)
    expand = (n,) + (1,) * len(P)
    sig_g   = np.asarray(sigmas, dtype=float).reshape(expand)
    costs_g = np.asarray(costs,  dtype=float).reshape(expand)
    X_g     = np.asarray(X,      dtype=float).reshape(expand)

    # Expand parameter arrays to (1, *P) so they broadcast with (n, *P)
    C_g  = C[None, ...]
    b_g   = b[None,  ...]

    # Threshold formula (your original + covariate term): 
    # costs + (sigmas^2 / tau^2) * (costs - mu_0) + b * X
    thresh = costs_g + b_g * X_g + C_g*sig_g 
    return thresh

def linear_shrink_constantK_threshold_fn(ratio,sigmas, cost):
    # Consider linear shrinkage to the grand mean where prior_scale controls the amount of
    # shrinkage
    thresh = cost + ratio[None, :] * (sigmas**2)[:, None]
    return thresh

def linear_shrinkage_threshold_fn(params, *, sigmas, costs):
    """
    params: array with shape (*P, 2), last axis = (tau_0, mu_0)
    returns: array with shape (n, *P)
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 2:
        raise ValueError("params[..., 2] must hold (tau_0, mu_0).")

    tau_0, mu_0 = np.moveaxis(params, -1, 0)   # each has shape (*P)

    # Shapes
    n = np.asarray(sigmas).shape[0]
    P = tau_0.shape  # tuple of grid dims

    # Expand sample-wise arrays to (n, *P)
    expand = (n,) + (1,) * len(P)
    sig_g   = np.asarray(sigmas, dtype=float).reshape(expand)
    costs_g = np.asarray(costs,  dtype=float).reshape(expand)

    # Expand parameter arrays to (1, *P) so they broadcast with (n, *P)
    tau_g = tau_0[None, ...]
    mu_g  = mu_0[None, ...]

    # Threshold formula (your original + covariate term): 
    # costs + (sigmas^2 / tau^2) * (costs - mu_0)
    thresh = costs_g + (sig_g**2) * (costs_g - mu_g) / (tau_g**2) 
    return thresh


def fh_threshold_fn(params, *, sigmas, costs, X, eps=0.0):
    """
    Thresholds of the form:
        delta_i = K_i + (sigma_i^2 / A) * (K_i - <X_i, beta>)

    Parameters
    ----------
    params : ndarray, shape (*P, 1 + p)
        Last axis packs (A, beta[0], ..., beta[p-1]) per grid point.
    sigmas : (n,)
    costs  : (n,)     # K_i
    X      : (n, p)

    Returns
    -------
    thresh : (n, *P)
    """
    params = np.asarray(params, dtype=float)
    X = np.asarray(X, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    costs = np.asarray(costs, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must have shape (n, p).")
    n, p = X.shape

    if params.shape[-1] != 1 + p:
        raise ValueError(f"params[..., 1+p] expected with p={p}; got last dim {params.shape[-1]}.")

    # Unpack (A, beta) from the last axis
    A = params[..., 0]           # (*P,)
    beta = params[..., 1:]       # (*P, p)

    # Grid shape (*P)
    P = A.shape
    expand_n = (n,) + (1,) * len(P)

    # Expand sample-wise arrays to (n, *P)
    sig_g   = sigmas.reshape(expand_n)
    costs_g = costs.reshape(expand_n)

    # Compute <X, beta> for every grid point using einsum:
    # beta_T has shape (p, *P); result is (n, *P)
    beta_T = np.moveaxis(beta, -1, 0)        # (p, *P)
    Xbeta  = np.einsum('np,p...->n...', X, beta_T)

    # Broadcast A to (n, *P)
    A_g = A[None, ...]

    if eps > 0:
        A_g = A_g + eps  # optional stabilization to avoid division by zero

    # Threshold
    thresh = costs_g + (sig_g**2) * (costs_g - Xbeta) / A_g
    return thresh



def t_stat_threshold_fn(params, *, sigmas, costs):
    """
    params: array with shape (*P, 1) -> (t_thresh,)
    returns: array with shape (n, *P)
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 1:
        raise ValueError("params[..., 1] must hold (t_thresh,).")

    (t_thresh,) = np.moveaxis(params, -1, 0)   # shape (*P)

    n = np.asarray(sigmas).shape[0]
    P = t_thresh.shape
    expand = (n,) + (1,) * len(P)

    sig_g   = np.asarray(sigmas, dtype=float).reshape(expand)
    costs_g = np.asarray(costs,  dtype=float).reshape(expand)
    t_g     = t_thresh[None, ...]               # shape (1, *P)

    return costs_g + t_g * sig_g   


def truncation_threshold_fn(params, *, costs):
    """
    params: array with shape (*P, 1) -> (thresh,)
    returns: array with shape (n, *P)
    NOTE: independent of per-sample arrays; broadcasts costs across n.
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 1:
        raise ValueError("params[..., 1] must hold (thresh,).")

    (thresh,) = np.moveaxis(params, -1, 0)      # shape (*P)

    n = np.asarray(costs).shape[0]
    P = thresh.shape
    expand_n = (n,) + (1,) * len(P)

    costs_g  = np.asarray(costs, dtype=float).reshape(expand_n)  # (n, *P)
    thresh_g = thresh[None, ...]                                 # (1, *P)

    return costs_g + thresh_g   


def close_gauss_threshold_fn(params, *, sigmas, costs, eps_log=1e-12):
    """
    Close-Gauss thresholds:
        delta_i = K_i + (sigma_i^2 / s0_i^2) * (K_i - m0_i)
        m0_i    = a1 + a2 * sigma_i
        s0_i^2  = exp(b1 + b2 * log(sigma_i))

    Parameters
    ----------
    params : ndarray, shape (*P, 5)
        Last axis packs (a1, a2, b1, b2).
    sigmas : (n,)
    costs  : (n,)          # K_i
    X      : (n,)
    eps_log : float        # floor for log stability

    Returns
    -------
    thresh : (n, *P)
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 4:
        raise ValueError("params[..., 5] must hold (a0, a1, a2, b1, b2).")

    a1, a2, b1, b2 = np.moveaxis(params, -1, 0)  # each is (*P)

    n = np.asarray(sigmas).shape[0]
    P = a1.shape
    expand_n = (n,) + (1,) * len(P)

    # Expand sample arrays to (n, *P)
    sig_g   = np.asarray(sigmas, dtype=float).reshape(expand_n)
    costs_g = np.asarray(costs,  dtype=float).reshape(expand_n)

    # Expand parameter arrays to (1, *P)
    a1_g = a1[None, ...]
    a2_g = a2[None, ...]
    b1_g = b1[None, ...]
    b2_g = b2[None, ...]

    # Components
    log_sig = np.log(np.maximum(sig_g, eps_log))
    m0 = a1_g + a2_g * sig_g
    s0_log = b1_g + b2_g * log_sig              # log s0^2
    shrink_ratio = (sig_g ** 2) * np.exp(-s0_log)  # sigma^2 / s0^2

    # Final threshold
    thresh = costs_g + shrink_ratio * (costs_g - m0)
    return thresh






def close_gauss_1dcvx_threshold_fn(params, *,sigmas, costs, X1, s0_sq, m0):
    """
    params: array with shape (*P, 1), last axis = (b1, )
    returns: array with shape (n, *P)
    """
    params = np.asarray(params, dtype=float)
    if params.shape[-1] != 1:
        raise ValueError("params[..., 1] must hold (b1,).")

    (b1,) = np.moveaxis(params, -1, 0)   # each has shape (*P)

    # Shapes
    n = np.asarray(costs).shape[0]
    P = b1.shape  # tuple of grid dims

    ratios = sigmas**2/s0_sq

    # Expand sample-wise arrays to (n, *P)
    expand = (n,) + (1,) * len(P)
    costs_g = np.asarray(costs,  dtype=float).reshape(expand)
    X1_g    = np.asarray(X1,      dtype=float).reshape(expand)
    ratios_g    = np.asarray(ratios,      dtype=float).reshape(expand)
    m0_g    = np.asarray(m0,      dtype=float).reshape(expand)
    

    # Expand parameter arrays to (1, *P) so they broadcast with (n, *P)
    b1_g  = b1[None, ...]

    thresh = (costs_g - b1_g * X1_g)/(1. - b1_g)*(1 + ratios_g) - m0_g*ratios_g

    return thresh


##### Helper Functions ######


def get_GLS_beta(Ys, Xs, sigmas):

    D = np.asarray(sigmas, dtype=float).reshape(-1)**2
    p = Xs.shape[1]

    #set_trace()
    Vinv = 1.0 / D
    XtVinv = (Xs.T * Vinv)     # p x m
    XtVinvX = XtVinv @ Xs      # p x p
    XtVinvY = XtVinv @ Ys      # p
    # Solve XtVinvX * beta = XtVinvY
    return np.linalg.solve(XtVinvX.reshape(p,p), XtVinvY.reshape(p,1))



