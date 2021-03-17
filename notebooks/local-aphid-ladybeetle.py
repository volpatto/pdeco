# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Aphid-Ladybeetle study

# +
import matplotlib.pyplot as plt
from numba import jit
import numpy as np  # linear algebra
from scipy.integrate import solve_ivp  # to solve ODE system
from scipy import optimize
import pandas as pd

np.seterr('raise')
# -

# ## Obtaining Initial Conditions
#
# We need to define Initial Conditions as functions in order to define them for each discretization point. Here we will fit ICs as polynomial functions.

# Loading data:

data_dir = "../data/simple/local/"
aphid_data = pd.read_excel(data_dir + 'aphid.xls')
ladybeetle_data = pd.read_excel(data_dir + 'ladybeetle.xls')

aphid_data

ladybeetle_data

# Retrieving IC data:

aphid_ic = aphid_data[aphid_data.time == 0].density.values[0]
ladybeetle_ic = ladybeetle_data[ladybeetle_data.time == 0].density.values[0]

aphid_ic

ladybeetle_ic

# +
y0_PPRM = aphid_ic, ladybeetle_ic

y0_PPRM

# +
import matplotlib.pyplot as plt

plt.plot(aphid_data.time.values, aphid_data.density.values, '-o')
plt.plot(ladybeetle_data.time.values, ladybeetle_data.density.values, '-o')
plt.xlabel('Time')
plt.ylabel('Population')
plt.show()


# -

# # Prey-Predator Rosenzweig-MacArthur model

# +
@jit(nopython=True)
def PPRM_model(
    t,
    X,
    r=1,
    K=10,
    a=1,
    h=1,
    ef=1,
    m=1,
):
    """
    Prey-Predator Rosenzweig-MacArthur (PPRM) python implementation.
    """
    u, v = X
    u_prime = r * u * ( 1 - u / K ) - a * u * v / ( 1 + a * h * u )
    v_prime = ef * a * u * v / ( 1 + a * h * u ) - m * v
    return u_prime, v_prime

def PPRM_ode_solver(
    y0,
    t_span,
    t_eval,
    r=1,
    K=10,
    a=1,
    h=1,
    ef=1,
    m=1,
):
    solution_ODE = solve_ivp(
        fun=PPRM_model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=(r, K, a, h, ef, m),
        method="LSODA",
    )
    return solution_ODE


# -

# * We now need to calibrate the parameters of the function. Firstly, we have to define a least-squares residual error function:

def PPRM_least_squares_error_ode(
    par, time_exp, f_exp, fitting_model, initial_conditions
):
    args = par
    f_exp1, f_exp2 = f_exp
    time_span = (time_exp.min(), time_exp.max())

    weighting_for_exp1_constraints = 1
    weighting_for_exp2_constraints = 1e1
    num_of_qoi = len(f_exp)

    try:
        y_model = fitting_model(initial_conditions, time_span, time_exp, *args)
        simulated_time = y_model.t
        simulated_ode_solution = y_model.y
        simulated_qoi1, simulated_qoi2 = simulated_ode_solution

        residual1 = f_exp1 - simulated_qoi1
        residual2 = f_exp2 - simulated_qoi2

        first_term = weighting_for_exp1_constraints * np.sum(residual1 ** 2.0)
        second_term = weighting_for_exp2_constraints * np.sum(residual2 ** 2.0)

        objective_function = 1 / num_of_qoi * (first_term + second_term)
    except ValueError:
        objective_function = 1e15

    return objective_function


# * Now we calibrate minimizing the residual applying the Differential Evolution method, a global optimization method, provided by `scipy`:

# +
def callback_de(xk, convergence):
    """
    This function is to show the optimization procedure progress.
    """
    print(f'parameters = {xk}\n')

bounds_PPRM = [
    (1e-10, 4),  # r
    (1e-10, 4),  # K
    (1e-10, 4),  # a
    (1e-10, 4),  # h
    (1e-10, 4),  # ef
    (1e-10, 4),  # m
]

result_PPRM = optimize.differential_evolution(
    PPRM_least_squares_error_ode,
    bounds=bounds_PPRM,
    args=(
        aphid_data.time.values,
        [aphid_data.density.values, ladybeetle_data.density.values],
        PPRM_ode_solver,
        y0_PPRM,
    ),
    popsize=30,
    strategy="best1bin",
    tol=1e-5,
    recombination=0.95,
    mutation=0.6,
    maxiter=2000,
    polish=True,
    disp=True,
    seed = 1234,  # for the sake of reproducibility
    callback=callback_de,
    workers=-1,
)

print(result_PPRM)
# -

# * Retrieving the calibrated parameter values:

# +
t0 = aphid_data.time.values.min()
tf = aphid_data.time.values.max()
days_to_forecast = 0
time_range = np.linspace(t0, tf + days_to_forecast, 100)
(
    r_deterministic,
    K_deterministic,
    a_deterministic,
    h_deterministic,
    ef_deterministic,
    m_deterministic,
) = result_PPRM.x

solution_ODE_PPRM = PPRM_ode_solver(
    y0_PPRM, 
    (t0, tf + days_to_forecast), 
    time_range, 
    *result_PPRM.x
)
t_computed_PPRM, y_computed_PPRM = solution_ODE_PPRM.t, solution_ODE_PPRM.y
u_PPRM, v_PPRM = y_computed_PPRM

parameters_dict = {
    "Model": "PPRM",
    u"$r$": r_deterministic,
    u"$K$": K_deterministic,
    u"$a$": a_deterministic,
    u"$h$": h_deterministic,
    u"$ef$": ef_deterministic,
    u"$m$": m_deterministic,
}

df_parameters_calibrated = pd.DataFrame.from_records([parameters_dict])
print(df_parameters_calibrated.to_latex(index=False))
# -

# #### Simulation

# +
import matplotlib.pyplot as plt

aphid_observed = aphid_data[:].copy()
ladybeetle_observed = ladybeetle_data[:].copy()

plt.plot(t_computed_PPRM, u_PPRM, '-x')
plt.plot(aphid_data.time.values, aphid_observed.density.values, 'o', label='Observed')

plt.xlabel('Time')
plt.ylabel('Aphid population')
plt.show()

plt.plot(t_computed_PPRM, v_PPRM, '-x')
plt.plot(ladybeetle_data.time.values, ladybeetle_observed.density.values, 'o', label='Observed')
plt.xlabel('Time')
plt.ylabel('Ladybeetle population')
plt.show()

# +
plt.figure(figsize=(9, 7))

plt.plot(
    aphid_data.time.values, aphid_data.density.values, label="Aphid", marker="s", linestyle="", markersize=10
)
plt.plot(t_computed_PPRM, u_PPRM, label=" ")

plt.plot(
    ladybeetle_data.time.values, ladybeetle_data.density.values, label="Ladybeetle", marker="v", linestyle="", markersize=10
)
plt.plot(t_computed_PPRM, v_PPRM, label=" ")

plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend(fancybox=True, shadow=True)
plt.grid()

plt.tight_layout()
plt.savefig("PPRM_deterministic_calibration.png")
plt.show()
