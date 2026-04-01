import numpy as np
from scipy.optimize import minimize_scalar
from solving_optimally_designed_signal import solve_optimally_designed_signal


def f_star(theta, ak):
    indices = [2 * k for k in range(len(ak))]
    theta = np.asarray(theta)
    M = np.vstack([np.cos(idx * theta) for idx in indices]).T
    return np.array((M.dot(ak))**2)


def iterative_refinement_scheme(theta_true, theta_hat_prev, r_k_prev, d_k_prev, q_factor, mk, zeta):

    d_k = q_factor * d_k_prev
    if d_k % 2 != 0:
        d_k += 1

    # Design signal using optimally designed signal 
    a_coeffs, Lk = solve_optimally_designed_signal(d_k, theta_hat_prev, r_k_prev)
    if a_coeffs is None:
        print("failed to find a solution.")
        return None, None, r_k_prev, Lk

    muk_true = f_star(theta_true, a_coeffs)[0]
    sigma_k_sq = np.abs(muk_true * (1 - muk_true))

    # Confidence interval for clipping
    I_prev = [theta_hat_prev - r_k_prev, theta_hat_prev + r_k_prev]
    j_vals = [f_star(I_prev[0], a_coeffs)[0], f_star(I_prev[1], a_coeffs)[0]]
    j_min, j_max = min(j_vals), max(j_vals)

    # Get theta_hat_k by simulating the measurement outcome with noise and applying the inverse mapping
    sk = muk_true + np.random.normal(0, np.sqrt(sigma_k_sq / mk))

    if sk < j_min:
        theta_hat_k = I_prev[0]
    elif sk > j_max:
        theta_hat_k = I_prev[1]
    else:
        obj = lambda t: (f_star(t, a_coeffs)[0] - sk)**2
        res = minimize_scalar(obj, bounds=(I_prev[0], I_prev[1]), method='bounded')
        theta_hat_k = res.x

    # Update radius
    error_k = np.abs(theta_hat_k - theta_true)
    r_k = 1 / (4 * d_k * q_factor * zeta)

    return theta_hat_k, error_k, r_k, Lk
