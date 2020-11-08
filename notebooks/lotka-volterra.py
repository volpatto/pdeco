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

# # Lotka-Volterra examples
#
# This notebook shows how to solve the following system:
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
# 1D and 2D cases are shown below.

# ## 1D case

# +
from pde import (PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid, 
                 CartesianGrid, MemoryStorage)
from pde import ExplicitSolver, ImplicitSolver, Controller, ScipySolver

# Species diffusivity coefficients (please be careful with this values)
Du = 1e-6
Dv = 1e-4  # predator assumed to be faster than prey

# Rate coefficients
r = 0.15
a = 0.2
b = 1.02  # Let's predator population increases when they prey because life is not easy
m = 0.05

# Functional response
f_function = f"+ {r} * u - {a} * u * v"  # don't forget to put + (or -) sign in the beginning
g_function = f"+ {b} * {a} * u * v - {m} * v"

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
num_points_in_x = 100
grid = CartesianGrid(bounds=[[0, 1]], shape=num_points_in_x)

# Initialize state (Initial Conditions)
u = ScalarField.from_expression(grid, "sin(pi * x)", label="Prey")
# v = ScalarField.from_expression(grid, "abs(sin(2 * pi * x))", label="Predator")
v = ScalarField.from_expression(grid, "0.1", label="Predator")
# v = ScalarField(grid, 0.1, label="Predator")
state = FieldCollection([u, v])  # state vector

# Define time tracker to plot and animate
x_axis_limits = (0, 1)
y_axis_limits = (0, 3.5)
tracker_plot_config = PlotTracker(show=True, plot_args={
        'ax_style': {'xlim': x_axis_limits, 'ylim': y_axis_limits},
    }
)
storage = MemoryStorage()
trackers = [
    "progress",  # show progress bar during simulation
    "steady_state",  # abort if steady state is reached
    storage.tracker(interval=1),  # store data every simulation time unit
    tracker_plot_config,  # show images during simulation
]

# Setup explicit solver
# explicit_solver = ExplicitSolver(eq)
# controller = Controller(explicit_solver, t_range=[0, 100], tracker=trackers)
# solve = controller.run(state, dt=1e-2)

# Setup scipy solver
scipy_solver = ScipySolver(eq, method='LSODA')
controller = Controller(scipy_solver, t_range=[0, 100], tracker=trackers)
solve = controller.run(state, dt=1e-2)

# +
u_storage = storage.extract_field(0)
v_storage = storage.extract_field(1)

u_storage.data

# +
from pde.visualization import movie

movie(v_storage, "v_field.mp4")
# -

# ## 2D case
#
# In this case, you just have to change the definition of your grid. Everything else remains the same.

# +
from pde import (PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid, 
                 CartesianGrid, MemoryStorage)
from pde import ExplicitSolver, ImplicitSolver, Controller

# Species diffusivity coefficients (please be careful with these values)
Du = 1e-6
Dv = 1e-4  # predator assumed to be faster than prey

# Rate coefficients
r = 0.15
a = 0.2
b = 1.02  # Let's predator population increases when they prey because life is not easy
m = 0.05

# Functional response
f_function = f"+ {r} * u - {a} * u * v"  # don't forget to put + (or -) sign in the beginning
g_function = f"+ {b} * {a} * u * v - {m} * v"

# (Dirichlet) Boundary condition example
bc_x = ({'value': 0}, {'value': 0})  # left right
bc_y = ({'value': 0}, {'value': 0})  # bottom top
bc = [bc_x, bc_y]

# Definition of PDE system
eq = PDE(
    {
        "u": f"{Du} * laplace(u)" + f_function,
        "v": f"{Dv} * laplace(v)" + g_function,
    },
    bc=bc
)

#################################################################
# Defining the mesh ** (HERE IS DIFFERENT COMPARED TO 1D CASE) **
num_points_in_x = 100
num_points_in_y = 100
mesh_shape = (num_points_in_x, num_points_in_y)
grid = CartesianGrid(bounds=[[0, 1], [0, 1]], shape=mesh_shape)
#################################################################

# Initialize state (Initial Conditions)
# u = ScalarField(grid, u_0, label="Prey")  # with this one you can give a value for all the domain
u = ScalarField.from_expression(grid, "sin(pi * x) * sin(pi * y)", label="Prey")
# v = ScalarField.from_expression(grid, "abs(sin(2 * pi * x) * sin(2 * pi * y))", label="Predator")
v = ScalarField(grid, 0.1, label="Predator")
state = FieldCollection([u, v])  # state vector

# Define time tracker to plot and animate
tracker_plot_config = PlotTracker(plot_args={
        "vmin": 0,  # min value in the color bar
        "vmax": 3   # max value in the color bar
    }
)
storage = MemoryStorage()
trackers = [
    "progress",  # show progress bar during simulation
    "steady_state",  # abort if steady state is reached
    storage.tracker(interval=1),  # store data every simulation time unit
    tracker_plot_config,  # show images during simulation
]

# Setup explicit solver
# explicit_solver = ExplicitSolver(eq)
# controller = Controller(explicit_solver, t_range=[0, 300], tracker=trackers)
# solve = controller.run(state, dt=1e-2)

# Setup implicit solver (if explicit does not work, try this one)
# implicit_solver = ImplicitSolver(eq)
# controller = Controller(implicit_solver, t_range=[0, 50], tracker=trackers)
# solve = controller.run(state)

# Setup scipy solver
scipy_solver = ScipySolver(eq, method='LSODA')
controller = Controller(scipy_solver, t_range=[0, 100], tracker=trackers)
solve = controller.run(state, dt=1e-2)

# print(controller.diagnostics)  # to debug

# +
u_storage = storage.extract_field(0)
v_storage = storage.extract_field(1)

grid.axes_coords
