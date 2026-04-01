## solver
import numpy as np
from gurobipy import Model, GRB, quicksum

def solve_optimally_designed_signal(d, theta_hat, r, N_amp=200, N_prior=100, N_alpha_grid=50, zeta=1):

    assert d % 2 == 0
    K_count = (d // 2) + 1
    indices = [2 * k for k in range(K_count)] 

    # Grids
    theta_amp = np.linspace(0, np.pi, N_amp)
    half_width = 1.0/(4*d * zeta)
    theta_prior = np.linspace(theta_hat - half_width, theta_hat + half_width, N_prior)
    alpha_grid = np.linspace(0.5, 1.0, N_alpha_grid)

    # Basis using 2*k
    A_amp = np.array([[np.cos(idx * t) for idx in indices] for t in theta_prior])
    A_deriv = np.array([[-idx * np.sin(idx * t) for idx in indices] for t in theta_prior])
    A_amp_full = np.array([[np.cos(idx * t) for idx in indices] for t in theta_amp])

    best_obj = -np.inf
    best_sol = None

    # Four sign cases: (s_g, s_gp) encodes the sign of g and g' on the prior interval.
    sign_cases = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # (s_g, s_gp)

    for alpha_val in alpha_grid:
        for s_g, s_gp in sign_cases:
            model = Model("alpha_beta_lp")
            model.setParam("OutputFlag", 0)

            # Variables: K_count cosine coefficients + beta
            a_vars = [model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"a_{idx}") for idx in indices]
            beta = model.addVar(lb=0.0, ub=float(d), name="beta")

            # L-infinity constraint: |g(theta)| <= 1 over the full amplitude grid
            for j in range(N_amp):
                expr = quicksum(A_amp_full[j, k] * a_vars[k] for k in range(K_count))
                model.addConstr(expr <= 1.0)
                model.addConstr(expr >= -1.0)

            # Sign-fixed constraints on the prior interval:
            #   s_g  * g(theta_i)  >= alpha   (enforces |g|  >= alpha with fixed sign)
            #   s_gp * g'(theta_i) >= beta    (enforces |g'| >= beta  with fixed sign)
            for i in range(N_prior):
                Fi_expr = quicksum(A_amp[i, k] * a_vars[k] for k in range(K_count))
                model.addConstr(s_g * Fi_expr >= alpha_val)

                Fpi_expr = quicksum(A_deriv[i, k] * a_vars[k] for k in range(K_count))
                model.addConstr(s_gp * Fpi_expr >= beta)

            model.setObjective(2.0 * alpha_val * beta, GRB.MAXIMIZE)
            model.optimize()

            if model.Status == GRB.OPTIMAL:
                if model.ObjVal > best_obj:
                    best_obj = model.ObjVal
                    best_sol = {
                        "a": {f"a_{indices[k]}": v.X for k, v in enumerate(a_vars)}
                    }

    # Scale down the signal for avoiding overshooting
    if best_sol is None:
        print('no solution from optimization')
        return None

    # Retrieve indices and coefficients
    indices = [2 * k for k in range((d // 2) + 1)]
    ak = np.array([best_sol["a"][f"a_{idx}"] for idx in indices])

    def g(theta,ak):
        theta = np.asarray(theta)
        # Reconstruct: sum a_k * cos(k * theta)
        M = np.vstack([np.cos(idx * theta) for idx in indices]).T
        return M.dot(ak)

    def gp(theta,ak):
        theta = np.asarray(theta)
        # Derivative: sum -a_k * k * sin(k * theta)
        M = np.vstack([-idx * np.sin(idx * theta) for idx in indices]).T
        return M.dot(ak)
    
    theta = np.linspace(0, np.pi, 3000) 
    theta_prior = np.linspace(theta_hat - r, theta_hat + r, 800) 

    g_vals = g(theta, ak)
    max_g = np.max(np.abs(g_vals))

    # Apply the safety scaling for QSP stability
    scale_factor = 0.999 / max_g if max_g > 1e-9 else 1.0 
    ak_scaled = ak * scale_factor

    # Results
    gp_vals = gp(theta,ak_scaled)
    g_prior_vals = g(theta_prior, ak_scaled)
    gp_prior_vals = gp(theta_prior, ak_scaled)
    dg2_vals = 2.0 * np.abs(g_vals * gp_vals)
    dg2_vals_prior = 2.0 * np.abs(g_prior_vals * gp_prior_vals)
    L = np.min(dg2_vals_prior)

    return ak_scaled, L



