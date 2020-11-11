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

# ## Obtaining Initial Conditions
#
# We need to define Initial Conditions as functions in order to define them for each discretization point. Here we will fit ICs as polynomial functions.

# Loading data:

# +
import pandas as pd

aphid_data = pd.read_csv('../data/aphid.csv')
ladybeetle_data = pd.read_csv('../data/ladybeetle.csv')
# -

aphid_data

# Retrieving IC data:

aphid_ic_data = aphid_data[aphid_data.time == 0].copy()
ladybeetle_ic_data = ladybeetle_data[ladybeetle_data.time == 0].copy()

aphid_ic_data

ladybeetle_ic_data

# ### Ladybird beetles IC

# For ladybird beetles is quite simple, we just need to take the mean:

# +
ladybeetle_ic = ladybeetle_ic_data.density.values.mean()

ladybeetle_ic
# -

# ### Aphids IC

# This is a more interesting case. Let's have a look on the points.

# +
import matplotlib.pyplot as plt

plt.plot(aphid_ic_data.x.values, aphid_ic_data.density.values, '-o')
plt.xlabel('x')
plt.ylabel('Population density')
plt.show()
# -

# It quite resembles a gaussian. So let us fit a gaussian to it.

# * Define a gaussian function:

# +
import numpy as np
from typing import Union

def gaussian(
    x: Union[float, np.ndarray],
    scale_term: float,
    mu: float, 
    sigma: float
) -> Union[float, np.ndarray]:
    """
    A univariate gaussian function.
    
    :param x:
        The value to evaluate the gaussian function.
        
    :param scale_term:
        The proportional term that multiplicates the gaussian exponential
        term.
        
    :param mu:
        The mean for the gaussian function.
        
    :param sigma:
        The standard deviation.
        
    :return:
        A value, or an array of values, computed with the gaussian function.
    """
    exponential_term = np.exp(- 1.0 / 2.0 * ((x - mu) / sigma) ** 2.0)
    return scale_term * exponential_term


# -

# * We now need to calibrate the parameters of the function. Firstly, we have to define a least-squares residual error function:

def calculate_least_squares_error(parameters, x_data, y_data, fitting_model):
    args = parameters
    y_model = fitting_model(x_data, *args)    
    residual = y_data - y_model
    return np.sum(residual ** 2.0)


# * Now we calibrate minimizing the residual applying the Differential Evolution method, a global optimization method, provided by `scipy`:

# +
from scipy import optimize


def callback_de(xk, convergence):
    """
    This function is to show the optimization procedure progress.
    """
    print(f'parameters = {xk}\n')
    

parameters_bounds = [(1e-5, 1000), (1e-5, 10), (1e-5, 1000)]
x_data = aphid_ic_data.x.values
y_data = aphid_ic_data.density.values
seed = 1234  # for the sake of reproducibility

result= optimize.differential_evolution(
    calculate_least_squares_error, 
    bounds=parameters_bounds, 
    args=(x_data, y_data, gaussian), 
    popsize=30,
    strategy='best1bin',
    tol=1e-8,
    recombination=0.7,
    maxiter=500,
    polish=True,
    disp=True,
    seed=seed,
    callback=callback_de,
)

print(result)
# -

# * Retrieving the calibrated parameter values:

# +
scale_term, mu, sigma = result.x

x = np.linspace(0, 9)
aphid_ic_points = gaussian(x, scale_term, mu, sigma)
# -

# * Plotting the fitted function to compare with data:

# +
plt.figure(figsize=(10,8))
plt.rcParams.update({'font.size': 16})

plt.plot(x, aphid_ic_points, '-', label='Fitted IC function')
plt.plot(aphid_ic_data.x.values, aphid_ic_data.density.values, '-o', label='Data')

plt.title('Aphid Initial Condition')
plt.xlabel('x')
plt.ylabel('Population density')

plt.grid(True)
plt.legend(fancybox=True, shadow=True)

plt.savefig('fitted_ic.png', dpi=300)
plt.show()
# -

# ## First model: classical Lotka-Volterra
#
# The problem consists in solving the system:
#
# \begin{equation}
# \begin{aligned}
# u_{t} &=D_{u} u_{x x}+f(u, v) \\
# v_{t} &=D_{v} v_{x x}+g(u, v)
# \end{aligned}
# \end{equation}
#
# in which $u$ is the prey and $v$ the predator, $D_u$ and $D_v$ are the diffusive coefficients, and $f(u, v)$ and $g(u, v)$ are the reactive terms that can have the following form:
#
# \begin{equation}
# \begin{array}{c}
# f(u, v)=r u-a u v \\
# g(u, v)=b a u v-m v
# \end{array}
# \end{equation}
#
# with the constants $r, a, b, m > 0$.
#

# ### Forward simulation

# +
from pde import (PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid, 
                 CartesianGrid, MemoryStorage)
from pde import ExplicitSolver, ImplicitSolver, Controller, ScipySolver

# Species diffusivity coefficients (please be careful with these values)
Du = 0.67e-3  # gathered from Banks
Dv = 0.21e-2  # gathered from Banks

# Banks functional response 1 (case A)
r1 = 0.136
r2 = 0.48e-3
p = 0.032
i = 11.0
e1 = 0.0012
e2 = 20.9
e3 = 0.009

# Functional response (case A)
f_function = f"+ {r1} * u - {r2} * u * u - {p} * u * v"  # don't forget to put + (or -) sign in the beginning
g_function = f"+ {i} - ({e1} + {e2} * exp(- {e3} * u)) * v"

# (Dirichlet) Boundary condition example
bc_left = {"value": 0.0}  # both unknowns are set to zero, unfortunately
bc_right = {"value": 0.0}
bc = [bc_left, bc_right]

# Definition of PDE system
eq = PDE(
    {
        "u": f"{Du} * laplace(u)" + f_function,
        "v": f"{Dv} * laplace(v)" + g_function,
    },
    bc=bc  # comment here if you want to "free" the boundaries
)

# Defining the mesh
x_min, x_max = 0, 9
dx = 0.2
num_points_in_x = int((x_max - x_min) / dx)
grid = CartesianGrid(bounds=[[x_min, x_max]], shape=num_points_in_x)

# Initialize state (Initial Conditions)
u = ScalarField.from_expression(
    grid, 
    f"{scale_term} * exp(- 1.0 / 2.0 * ((x - {mu}) / {sigma}) ** 2.0)", 
    label="Prey"
)
v = ScalarField(grid, ladybeetle_ic, label="Predator")
state = FieldCollection([u, v])  # state vector

# Define time tracker to plot and animate
x_axis_limits = (x_min, x_max)
y_axis_limits = (0, 650)
tracker_plot_config = PlotTracker(show=True, plot_args={
        'ax_style': {'xlim': x_axis_limits, 'ylim': y_axis_limits},
    }
)
storage = MemoryStorage()
dt = 1e-2
trackers = [
    "progress",  # show progress bar during simulation
    "steady_state",  # abort if steady state is reached
    storage.tracker(interval=0.25),  # store data every simulation time unit
    tracker_plot_config,  # show images during simulation
]

# Select backend solver
# solver = ExplicitSolver(eq)  # Built-in explicit solver
solver = ScipySolver(eq, method='LSODA')  # SciPy solver

# Setup solver
controller = Controller(solver, t_range=[0, 2], tracker=trackers)
solve = controller.run(state, dt=dt)
# -

# Retrieving data stored spaced by 1.0 as time-records:

# +
u_storage = storage.extract_field(0)
v_storage = storage.extract_field(1)

u_storage.data
# -

# Retrieving mesh coordinates:

# +
x_points = grid.axes_coords[0]

x_points
# -

# ### Comparing simulation with measurements

# +
# time_labels_dict = {
#     0: 0,
#     1: 2,
#     2: 3
# }

time_labels_dict = {
    0: 0,
    1: 1,
    4: 2,
    8: 3
}

# +
time_indices_to_plot = list(time_labels_dict.keys())

time_indices_to_plot
# -

# #### Aphids

for time_index, aphid_simulation in enumerate(u_storage.data):
    if time_index in time_indices_to_plot:
        plt.figure(figsize=(8, 6))
        plt.ylim([0, 650])

        idx_for_observed = time_labels_dict[time_index]  # we skip time 1/4
        aphid_observed = aphid_data[aphid_data.time == idx_for_observed].copy()
        plt.plot(aphid_observed.x.values, aphid_observed.density.values, 'o', label='Observed')
        plt.plot(x_points, aphid_simulation, '-', label='Simulated')

        plt.xlabel('x')
        plt.ylabel('Population density')
        plt.title(f'Aphid at time {idx_for_observed}')

        plt.grid(True)
        plt.legend(shadow=True)
        plt.show()

# #### Ladybird beetles

for time_index, ladybeetle_simulation in enumerate(v_storage.data):
    if time_index in time_indices_to_plot:
        plt.figure(figsize=(8, 6))
        plt.ylim([0, 13])

        idx_for_observed = time_labels_dict[time_index]  # we skip time 1/4
        ladybeetle_observed = ladybeetle_data[ladybeetle_data.time == idx_for_observed].copy()
        plt.plot(ladybeetle_observed.x.values, ladybeetle_observed.density.values, 'o', label='Observed')
        plt.plot(x_points, ladybeetle_simulation, '-', label='Simulated')

        plt.xlabel('x')
        plt.ylabel('Population density')
        plt.title(f'Ladybird beetle at time {idx_for_observed}')

        plt.grid(True)
        plt.legend(shadow=True)
        plt.show()
