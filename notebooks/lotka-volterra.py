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
# -

import numpy as np
t_recorded = storage.times  # retrieve all recorded simulation times
x_idx = 30
u_storage_array = np.array(u_storage.data)  # convert list of arrays to a 2D array (to use 2D index)
u_at_given_x = u_storage_array[:, x_idx]  # [t_idx, x_idx], retrieve all times (:) for a given x_idx
u_at_given_x

# +
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(t_recorded, u_at_given_x, "-x")
plt.xlabel("Time")
plt.ylabel(fr"$u$ at x = {u_storage.grid.axes_coords[0][x_idx]}")
plt.grid()
plt.show()

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
# -

# ## Handling simulation outcomes (post-processing)

# ### Retrieving the grid coordinates

grid = storage.grid

# * x-axis

grid.axes_coords[0]

# * y-axis

grid.axes_coords[1]

# * Both (a list of arrays)

grid.axes_coords

# ### Retrieving the fields' values

# * Field `u`

u_storage = storage.extract_field(0)
u_storage.data

# * Field `v`

v_storage = storage.extract_field(1)
v_storage.data

len(v_storage.data)

v_storage.data[0]

# ### Solution at a given time index
#
# It can be simply accessed by the index value. Suppose we want time `12` of the simulation (note that this is not the simulation time) for `u` field. We can do:

u_storage.data[12]

# Thus, `u_storage.data` is a list indexed by the time. So, `u_storage.data[t_idx]` returns a 2D array (for the 2D problem) containing the solution at time index `t_idx`.

# ### Solution at a given time
#
# To retrieve the solution at a given time, you must know the discrete time points beforehand. For instance, let's consider the 2D LV simulation as before. We know that $t = [0, 100]$ with `storage.tracker(interval=1)`. It means that the simulation result is recorded with simulation time interval of 1. So we have the solution for times $1, 2, 3, \ldots, 100$. However, note that **this is not** the time increment in the simulation, which is configured in `solve = controller.run(state, dt=1e-2)`. Hence, $\Delta t = 0.01$. To have a more interesting case, let us run the simulation with a different setup.

# +
tracker_plot_config = PlotTracker(plot_args={
        "vmin": 0,  # min value in the color bar
        "vmax": 3   # max value in the color bar
    }
)
storage = MemoryStorage()
trackers = [
    "progress",
    "steady_state",
    storage.tracker(interval=0.5),  # note that we are changing this config parameter
    tracker_plot_config,
]

# Setup scipy solver
scipy_solver = ScipySolver(eq, method='LSODA')
controller = Controller(scipy_solver, t_range=[0, 100], tracker=trackers)
solve = controller.run(state, dt=1e-2)
# -

u_storage = storage.extract_field(0)
len(u_storage.data)

# Thus, we have 201 recorded simulation time, the double of time records we had before. Suppose that we want the simulation at time 16.5. It can be easily calculate with $t = t_0 + t_{\text{idx}}\times \Delta_i t \to t_{\text{idx}} = \frac{(t - t_0)}{\Delta_i t}$, where $\Delta_i t$ is the time record increment. We know that $t = 16.5$, $t_0 = 0$ and $\Delta_i t = 0.5$. Thus,

# +
t_0 = 0
t_desired = 16.5
deltai_t = 0.5
t_idx = int((t_desired - t_0) / deltai_t)

t_idx
# -

# Hence, the `u` field solution at $t = 16.5$ is

u_storage.data[t_idx]

# ### Solution at a given coordinate
#
# Suppose that, for a given time $t_i$, we want all the field `u` values at a given `x` in the `x-axis` range. This problem is quite similar to the previous one. We know the range of `x-axis`, which is defined in `grid = CartesianGrid(bounds=[[0, 1], [0, 1]], shape=mesh_shape)`, so `x_range = [0, 1]`. We also know the number of points on the `x-axis`, which we defined in `num_points_in_x = 100`. However, let's have a look of how discrete points are distribute on `x-axis`:

storage.grid.axes_coords[0]

# Interesting... we have set that we wanted `x_range = [0, 1]` with `dx = (x_final - x_initial) / num_points_in_x`, thus `dx = 0.01`. So why we do not have `x_points = array([0.00, 0.01, 0.02, 0.03, ..., 0.98, 0.99, 1.00])`? The answer is: `py-pde` was designed following a "cell" discretization fashion, see the figure below:

# ![](img/discretization_cropped.svg)

# It means that given an interval $[x_{min}, x_{max}]$, it will generate N points on the required axis/direction depending on your problem (and coordinate frame system). Of note, at the moment only uniform grids are supported by `py-pde`, so the cell/points spacing will be constant through its axis. Thus, if you want to retrieve the point index at a given axis, you can use:
#
# \begin{equation}
# \begin{aligned}
# &x_{i} =x_{\min }+\left(i+\frac{1}{2}\right) \Delta x \text { for } i=0, \ldots, N-1 \\
# &\Delta x =\frac{x_{\max }-x_{\min }}{N}
# \end{aligned}
# \end{equation}
#
# Suppose we want the values of `u` field for a fixed `x_i = 0.505` and `t = 16.5`. So we can calculate the required index as:
#
# \begin{equation}
# i = \frac{x_i - x_{min}}{\Delta x} - \frac{1}{2}
# \end{equation}
#
# Hence:

delta_x = 0.01
x_i = 0.505
x_min = 0
i = int((x_i - x_min) / delta_x - 1 / 2)
i

# We can check if this value is correct:

x_grid = storage.grid.axes_coords[0]
x_grid[i]

# So, the field `u` at the required time and space is:

u_storage.data[t_idx][i, :]

# You can easily plot the values with `matplotlib`:

# +
import matplotlib.pyplot as plt

y_grid = storage.grid.axes_coords[1]
u_solution_at_given_x = u_storage.data[t_idx][i, :]

plt.figure(figsize=(8, 6))
plt.plot(y_grid, u_solution_at_given_x, "-x")
plt.xlabel("y-axis")
plt.ylabel("Field u")
plt.title(fr"Field $u$ at x = {x_i} and t = {t_desired}")
plt.grid()
plt.show()


# -

# However, such way of generate a grid can present some inconvenience. For instance, consider the case you want the solution at `x = 0.5`. How you should discretize `x-axis`? The solution for this problem is unknown as far as I know. But... we have a workaround. Let's define the following utility function:

def find_idx_nearest_value_in_array(array, value):
    import numpy as np
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


# With this function, we can retrieve the array index corresponding to the near array element to a given value. Let's see it in action. Suppose, again, that we want to know the index corresponding to `x = 0.5` in our grid defined on the `x-axis`. We can do now:

target_idx, closest_x_grid_value = find_idx_nearest_value_in_array(x_grid, 0.5)
print(f"idx = {target_idx}; closest value = {closest_x_grid_value}")


# With `target_idx`, we can proceed as we did before for `x = 0.505` above. A similar utility function could be implemented using Python `int` built-in function with truncation, as below:

def find_idx_truncanted_value_in_array(x_desired, x_min, delta):
    return int((x_desired - x_min) / delta - 1 / 2)


target_idx_2 = find_idx_truncanted_value_in_array(0.5, x_min, delta_x)
print(f"idx = {target_idx_2}; closest value = {x_grid[target_idx_2]}")

# Note that the results from such utility functions can differ with each other.
