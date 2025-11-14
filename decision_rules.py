import methods  
import numpy as np
from scipy.stats import norm

# 1. Naive p-value
def get_naive_pval_decisions(Ys, sigmas, costs):
    return (Ys - costs >= norm(0,1).ppf(0.95) * sigmas).astype(int)

# 2. ASSURE p-value
def get_assure_pval_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.t_stat_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    beta_opt = param_grid[np.argmax(welfare)]
    decisions = (Ys - costs >= beta_opt * sigmas).astype(int)
    return decisions


# 3. Coupled Bootstrap p-value

def get_cb_pval_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.t_stat_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    beta_opt = param_grid[np.argmax(welfare)]
    decisions = (Ys - costs >= beta_opt * sigmas).astype(int)
    return decisions

# 4. Naive Truncation
def get_naive_truncation_decisions(Ys, sigmas, costs):
    return (Ys - costs >= 0).astype(int)


# 5. ASSURE Truncation
def get_assure_truncation_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.truncation_threshold_fn(params,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    beta_opt = param_grid[np.argmax(welfare)]
    decisions = (Ys - costs >= beta_opt).astype(int)
    return decisions

# 6. CB Truncation
def get_cb_truncation_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.truncation_threshold_fn(params,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    beta_opt = param_grid[np.argmax(welfare)]
    decisions = (Ys - costs >= beta_opt).astype(int)
    return decisions

###################################
#### Linear Shrinkage
###################################


def constant_cost_linear_shrinkage(Ys, sigmas, cost, params, function):
    
    welfare, welfare_se = function(
        Ys,
        sigmas,
        lambda params: methods.linear_shrink_constantK_threshold_fn(params, sigmas, cost=cost),
        params,
        costs=cost,
    )

    beta_opt = params[np.argmax(welfare)]
    decisions = (Ys >= cost + beta_opt*sigmas**2)
    return decisions.astype(int), welfare, welfare_se




# 7. Plug-in Linear Shrinkage 
def get_naive_linear_shrinkage_decisions(Ys, sigmas, costs):

    grand_mean = np.mean(Ys)
    grand_var = np.var(Ys) - np.mean(sigmas**2)
    thresholds = costs + (sigmas**2)*(costs - grand_mean)/grand_var

    return (Ys >= thresholds).astype(int)




# 8. ASSURE Linear Shrinkage
def get_assure_linear_shrinkage_decisions(Ys, sigmas, costs, param_grid):
    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.linear_shrinkage_threshold_fn(params,
                                    sigmas=sigmas,
                                    costs=costs),
        param_grid,
        costs=costs,
    )

    tau_opt, mu_opt = param_grid[np.argmax(welfare),:]

    thresholds = costs + (sigmas**2)*(costs - mu_opt)/tau_opt**2
    decisions = (Ys >= thresholds).astype(int)

    return decisions

# 9. Coupled Bootstrap
def get_cb_linear_shrinkage_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.linear_shrinkage_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    tau_opt, mu_opt = param_grid[np.argmax(welfare),:]
    opt_params = (tau_opt, mu_opt)

    thresholds = costs + (sigmas**2)*(costs - mu_opt)/tau_opt**2

    decisions = (Ys >= thresholds).astype(int)

    return decisions

# 10. NPMLE Decisions
def get_npmle_decisions(Ys, sigmas, costs):
    posterior_means, prior = methods.npmle_gaussian_hetero_rebayes(Ys, sigmas)
    
    decisions = (posterior_means >= costs).astype(int)
    return decisions, posterior_means

# 11. CLOSE NPMLE
def get_CLOSE_npmle_decisions(Ys, sigmas, costs):
    posterior_means = methods.CLOSE(Ys, sigmas)[1]
    
    decisions = (posterior_means >= costs).astype(int)
    return decisions, posterior_means

# 11. Linear Threshold
def get_assure_linear_decisions(Ys, sigmas, costs, Xs, param_grid):

    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.linear_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs,
                                                    X=Xs),
        param_grid,
        costs=costs,
    )

    C_opt, b_opt = param_grid[np.argmax(welfare),:]
    opt_params = (C_opt, b_opt)

    thresholds = costs + b_opt*Xs + sigmas*C_opt
    decisions = (Ys >= thresholds).astype(int)

    return decisions

# 11. Linear Threshold
def get_cb_linear_decisions(Ys, sigmas, costs, Xs, param_grid):

    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.linear_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs,
                                                    X=Xs),
        param_grid,
        costs=costs,
    )

    C_opt, b_opt = param_grid[np.argmax(welfare),:]
    opt_params = (C_opt, b_opt)

    thresholds = costs + b_opt*Xs + sigmas*C_opt
    decisions = (Ys >= thresholds).astype(int)

    return decisions

# 12. CLOSE Gauss Threshold

def get_close_gauss_decisions(Ys, sigmas, costs, param_grid):

    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.close_gauss_threshold_fn(params,
                                                    sigmas=sigmas,
                                                    costs=costs),
        param_grid,
        costs=costs,
    )

    opt_params = param_grid[np.argmax(welfare),:]
    a1_opt, a2_opt, b1_opt, b2_opt = opt_params

    log_sig = np.log(sigmas)
    m0 = a1_opt + a2_opt * sigmas
    s0_sq_log = b1_opt + b2_opt * log_sig              # log s0^2
    shrink_ratio = (sigmas ** 2) * np.exp(-s0_sq_log)  # sigma^2 / s0^2

    # Final threshold
    thresholds = costs + shrink_ratio * (costs - m0)

    decisions = (Ys >= thresholds).astype(int)

    return decisions



###############
################
#Fay Heriott Class
################
################

def get_fay_heriott_decisions(Ys, sigmas, costs, Xs):
    """
    Fay-Herriot area-level EB shrinkage with A estimated by the FH moment equation (Eq. 2.21).

    Parameters
    ----------
    Ys : array-like, shape (m,)
        Direct estimators Y_i for each area.
    sigmas : array-like, shape (m,)
        Known standard errors for the direct estimators. 
    Xs : array-like, shape (m, p)
        covariates/design matrix X.
    """

    res = methods.fay_heriott(Ys, sigmas, Xs)
    posterior_means = res["theta_hat"]
    decisions = (posterior_means > costs).astype(int)
    return decisions, posterior_means



def get_assure_fh_decisions(Ys, sigmas, costs, Xs, param_grid):
    """
    Xs : (m,p)

    beta_opt : (p,1)
    """
    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.fh_threshold_fn(params,
                                    sigmas=sigmas,
                                    costs=costs,
                                    X = Xs),
        param_grid,
        costs=costs,
    )

    theta_opt = param_grid[np.argmax(welfare),:]

    A_opt    = float(theta_opt[0])            # scalar
    beta_opt = np.asarray(theta_opt[1:], float).reshape(-1)  # (p,)

    # 3) compute thresholds with 1-D shapes
    xb = Xs @ beta_opt                        # (n,)
    thresholds = costs + (sigmas**2) * (costs - xb) / A_opt  # (n,)
    decisions  = (Ys >= thresholds).astype(int)              # (n,)

    return decisions


def get_cb_fh_decisions(Ys, sigmas, costs, Xs, param_grid):
    """
    Xs : (m,p)

    beta_opt : (p,1)
    """
    welfare, _ = methods.coupled_bootstrap(
        Ys,
        sigmas,
        lambda params: methods.fh_threshold_fn(params,
                                    sigmas=sigmas,
                                    costs=costs,
                                    X = Xs),
        param_grid,
        costs=costs,
    )

    theta_opt = param_grid[np.argmax(welfare),:]

    A_opt    = float(theta_opt[0])            # scalar
    beta_opt = np.asarray(theta_opt[1:], float).reshape(-1)  # (p,)

    # 3) compute thresholds with 1-D shapes
    xb = Xs @ beta_opt                        # (n,)
    thresholds = costs + (sigmas**2) * (costs - xb) / A_opt  # (n,)
    decisions  = (Ys >= thresholds).astype(int)              # (n,)

    return decisions




def get_assure_1dcvx_combination_decisions(Ys, sigmas, costs, X1, s0_sq, m0, param_grid):
    welfare, _ = methods.assure(
        Ys,
        sigmas,
        lambda params: methods.close_gauss_1dcvx_threshold_fn(params,
                                    sigmas=sigmas,
                                    costs=costs,
                                    X1 = X1,
                                    s0_sq=s0_sq,
                                    m0=m0),
        param_grid,
        costs=costs,
    )

    b1_opt = param_grid[np.argmax(welfare),:]
    ratios = sigmas**2/s0_sq

    thresholds = (costs - b1_opt * X1 )/(1. - b1_opt)*(1 + ratios) - m0*ratios
    decisions = (Ys >= thresholds).astype(int)

    return decisions, welfare, b1_opt