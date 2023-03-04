import numpy as np
import pysindy_local as ps
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct, ConstantKernel
from ModelPlots import ModelPlots
from ModelCalibration import ModelCalibration
from ModelSelection import ModelSelection
from DataDenoising import DataDenoising

def is_new_model(model_set, model, n_vars, precision):
	for model_element in model_set:
		flag = True
		for i in range(n_vars):
			if model_element.equations(precision = precision)[i] != model.equations(precision = precision)[i]:
				flag = False
				break

		if flag:
			return False

	return True

experiment_id = 0

# Method parameters
fd_order = 2
poly_degrees = range(2, 4)
fourier_nfreqs = range(1, 2)
optimizer_method = "STLSQ+SA"
precision = 3

plot_sse = True
plot_qoi = True
plot_musig = False
plot_simulation = False
calibration_mode = None

stlsq_alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

# Read train data
data = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t = data["Time"]
X_aphid = data["Aphid"].reshape((-1, 1))
X_ladybeetle = data["Ladybeetle"].reshape((-1, 1))

dd = DataDenoising(X_aphid, t, ["x"])
dd.plot_sma([5, 10, 20])
dd.plot_ema([0.1, 0.2, 0.3], [False])
dd.plot_l2r([10.0, 100.0, 1000.0])
dd.plot_tvr([0.001], [0.25, 0.5, 1.0])
# dd.plot_gpr([RBF(length_scale = 10, length_scale_bounds = "fixed"),
# 	RBF(length_scale = 20, length_scale_bounds = "fixed"),
# 	RBF(length_scale = 30, length_scale_bounds = "fixed")], 
# 	[1], [1.0e-10], ["RBF(10)", "RBF(20)", "RBF(30)"], 0
# )
# dd.plot_gpr([ExpSineSquared()*Matern(),
# 	RBF(),
# 	Matern()],
# 	[1], [1.0e-10], ["ExpSineSquared*Matern", "RBF", "Matern"], 1
# )
# dd.plot_gpr([RationalQuadratic(),
# 	ExpSineSquared(),
# 	DotProduct()],
# 	[1], [1.0e-10], ["RationalQuadratic", "ExpSineSquared", "DotProduct"], 2
# )
dd.plot_gpr([RBF(length_scale_bounds = (20, 40)),
	ConstantKernel(constant_value_bounds = (1.0e-7, 1.0e-6))*RBF(length_scale_bounds = (20, 40)),
	ConstantKernel(constant_value_bounds = (1.0e-7, 1.0e-6))*RBF(length_scale_bounds = (20, 40)) + WhiteKernel(noise_level_bounds = (1.0e-10, 1.0e-9))],
	[50], [1.0e-10], ["RBF", "ConstantKernel*RBF", "ConstantKernel*RBF + WhiteKernel"], 3
)
t_pred = np.linspace(t[0], t[-1], 200)
X_aphid, X_aphid_min, X_aphid_max = dd.gaussian_process_regression(
	ConstantKernel(constant_value_bounds = (1.0e-7, 1.0e-6))*RBF(length_scale_bounds = (20, 40)) + WhiteKernel(noise_level_bounds = (1.0e-10, 1.0e-9)), 
	50, 1.0e-10, t_pred
)
# X_aphid, X_aphid_min, X_aphid_max = dd.gaussian_process_regression(
# 	RBF(length_scale_bounds = (10, 100)) + WhiteKernel(noise_level_bounds = (1e-10, 5e-10)), 
# 	100, 1.0e-10, t_pred
# )

dd = DataDenoising(X_ladybeetle, t, ["y"])
dd.plot_sma([5, 10, 20])
dd.plot_ema([0.1, 0.2, 0.3], [False])
dd.plot_l2r([10.0, 100.0, 1000.0])
dd.plot_tvr([0.001], [0.25, 0.5, 1.0])
# dd.plot_gpr([RBF(length_scale = 10, length_scale_bounds = "fixed"),
# 	RBF(length_scale = 20, length_scale_bounds = "fixed"),
# 	RBF(length_scale = 30, length_scale_bounds = "fixed")], 
# 	[1], [1.0e-10], ["RBF(10)", "RBF(20)", "RBF(30)"], 0
# )
# dd.plot_gpr([ExpSineSquared()*Matern(),
# 	RBF(),
# 	Matern()],
# 	[1], [1.0e-10], ["ExpSineSquared*Matern", "RBF", "Matern"], 1
# )
# dd.plot_gpr([RationalQuadratic(),
# 	ExpSineSquared(),
# 	DotProduct()],
# 	[1], [1.0e-10], ["RationalQuadratic", "ExpSineSquared", "DotProduct"], 2
# )
dd.plot_gpr([RBF(length_scale_bounds = (120, 140)),
	ConstantKernel(constant_value_bounds = (1.0, 3.0))*RBF(length_scale_bounds = (120, 140)),
	ConstantKernel(constant_value_bounds = (60.0, 70.0))*RBF(length_scale_bounds = (25, 45)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2))],
	[50], [1.0e-10], ["RBF", "ConstantKernel*RBF", "ConstantKernel*RBF + WhiteKernel"], 3
)
t_pred = np.linspace(t[0], t[-1], 200)
# X_ladybeetle, X_ladybeetle_min, X_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (60.0, 70.0))*RBF(length_scale_bounds = (25, 45)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_pred
# )
X_ladybeetle, X_ladybeetle_min, X_ladybeetle_max = dd.gaussian_process_regression(
	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
	50, 1.0e-10, t_pred
)
# X_ladybeetle, X_ladybeetle_min, X_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0))*RBF(length_scale_bounds = (20, 45)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_pred
# )
# X_ladybeetle, X_ladybeetle_min, X_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_pred
# )

t = t_pred
X = np.hstack((X_aphid, X_ladybeetle))

# X_cs = CubicSpline(t, X)

# t = np.linspace(t[0], t[-1], 100)
# X = X_cs(t)

X0 = X[0, :]
t_steps = len(t)

# Read test data
data_test = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t_test = data_test["Time"]
X_test_aphid = data_test["Aphid"].reshape((-1, 1))
X_test_ladybeetle = data_test["Ladybeetle"].reshape((-1, 1))

dd = DataDenoising(X_test_aphid, t_test, ["x"])
t_test_pred = np.linspace(t_test[0], t_test[-1], 200)
# X_test_aphid, X_test_aphid_min, X_test_aphid_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (1.0e-7, 1.0e-6))*RBF(length_scale_bounds = (20, 40)) + WhiteKernel(noise_level_bounds = (1.0e-10, 1.0e-9)), 
# 	50, 1.0e-10, t_test_pred
# )
X_test_aphid, X_test_aphid_min, X_test_aphid_max = dd.gaussian_process_regression(
	RBF(length_scale_bounds = (10, 100)) + WhiteKernel(noise_level_bounds = (1e-10, 5e-10)), 
	100, 1.0e-10, t_test_pred
)

dd = DataDenoising(X_test_ladybeetle, t_test, ["y"])
t_test_pred = np.linspace(t_test[0], t_test[-1], 200)
# X_test_ladybeetle, X_test_ladybeetle_min, X_test_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (60.0, 70.0))*RBF(length_scale_bounds = (25, 45)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_test_pred
# )
# X_test_ladybeetle, X_test_ladybeetle_min, X_test_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_test_pred
# )
# X_test_ladybeetle, X_test_ladybeetle_min, X_test_ladybeetle_max = dd.gaussian_process_regression(
# 	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0))*RBF(length_scale_bounds = (20, 45)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
# 	50, 1.0e-10, t_test_pred
# )
X_test_ladybeetle, X_test_ladybeetle_min, X_test_ladybeetle_max = dd.gaussian_process_regression(
	ConstantKernel(constant_value_bounds = (1.0e-1, 1.0)) + WhiteKernel(noise_level_bounds = (1.0e-3, 1.0e-2)), 
	50, 1.0e-10, t_test_pred
)

t_test = t_test_pred
X_test = np.hstack((X_test_aphid, X_test_ladybeetle))

# X_test_cs = CubicSpline(t_test, X_test)

# t_test = np.linspace(t_test[0], t_test[-1], 100)
# X_test = X_test_cs(t_test)

X0_test = X_test[0, :]

model_set = []
for poly_degree in poly_degrees:
	for fourier_nfreq in fourier_nfreqs:
		for stlsq_alpha in stlsq_alphas:
			experiment_id += 1
			print("Experimento " + str(experiment_id) 
				+ ": Grau = " + str(poly_degree) 
				+ ", Frequência = " + str(fourier_nfreq) 
				+ ", alpha = " + str(stlsq_alpha) + "\n"
			)

			# Define method properties
			differentiation_method = ps.FiniteDifference(order = fd_order)
			# differentiation_method = ps.SmoothedFiniteDifference()
			feature_library = ps.PolynomialLibrary(degree = poly_degree) # + ps.FourierLibrary(n_frequencies = fourier_nfreq)
			optimizer = ps.STLSQ(
				alpha = stlsq_alpha,
				fit_intercept = False,
				verbose = True,
				window = 3,
				epsilon = 1.0,
				time = t,
				sa_times = np.array([10.0, 20.0, 30.0, 40.0]),
				non_physical_features = [['1', 'y', 'y^2', 'y^3'],
					['1', 'x', 'x^2', 'x^3']
				]
			)

			# Compute sparse regression
			model = ps.SINDy(
				differentiation_method = differentiation_method,
				feature_library = feature_library,
				optimizer = optimizer,
				feature_names = ["x", "y"]
			)
			model.fit(X, t = t)
			model.print(precision = precision)
			print("\n")

			# Generate model plots
			mp = ModelPlots(model, optimizer_method, experiment_id)
			if plot_sse:
				mp.plot_sse()
			if plot_qoi:
				mp.plot_qoi()
			if plot_musig:
				mp.plot_musig()
			if plot_simulation:
				mp.plot_simulation(X, t, X0, precision = precision)

			# Add model to the set of models
			if not model_set or is_new_model(model_set, model, len(model.feature_names), precision):
				model_set.append(model)

# Compute number of terms
ms = ModelSelection(model_set, t_steps)
ms.compute_k()

for model_id, model in enumerate(model_set):
	print("Modelo " + str(model_id+1) + "\n")
	model.print(precision = precision)
	print("\n")

	dd = DataDenoising(X_test, t_test, model.feature_names)

	# Compute derivative
	X_dot_test = model.differentiate(X_test, t_test)
	dd.plot_derivative(X_dot_test, t_test, 0, X0_test)

	# Simulate with another initial condition
	# if calibration_mode is None:
	# 	simulation = model.simulate(X0_test, t = t_test)
	# elif calibration_mode == "LM":
	# 	mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
	# 	mc.levenberg_marquardt()
	# 	model.print(precision = precision)
	# 	print("\n")

	# 	simulation = model.simulate(X0_test, t = t_test)
	# elif calibration_mode == "Bayes":
	# 	mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
	# 	mc.bayesian_calibration()
	# 	mc.traceplot()
	# 	mc.plot_posterior()
	# 	mc.plot_pair()
	# 	X0_test = mc.summary()
	# 	print("\n")
	# 	model.print(precision = precision)
	# 	print("\n")

	# 	simulation, simulation_min, simulation_max = mc.get_simulation()

	if model_id == 2:
		mc = ModelCalibration(model, model_id, X_test, t, X0_test, 0)
		mc.bayesian_calibration()
		mc.traceplot()
		mc.plot_posterior()
		mc.plot_pair()
		X0_test = mc.summary()
		print("\n")
		model.print(precision = precision)
		print("\n")

		simulation, simulation_min, simulation_max = mc.get_simulation()
	else:
		simulation = model.simulate(X0_test, t = t_test)

	# Generate figures
	fig, ax1 = plt.subplots(1, 1, figsize = (15, 7.5), dpi = 300)
	ax2 = ax1.twinx()
	ax1.plot(t_test, X_test[:,0], "ko", alpha = 0.5, markersize = 3)
	ax2.plot(t_test, X_test[:,1], "k^", alpha = 0.5, markersize = 3)
	ax1.plot(t_test, simulation[:,0], "b", alpha = 1.0, linewidth = 1)
	ax2.plot(t_test, simulation[:,1], "g", alpha = 1.0, linewidth = 1)
	if model_id == 2:
		ax1.fill_between(t, simulation_min[:,0], simulation_max[:,0], color = "b", alpha = 0.4)
		ax2.fill_between(t, simulation_min[:,1], simulation_max[:,1], color = "g", alpha = 0.4)
	ax1.set_xlabel(r"Time $t$")
	ax1.set_ylabel(r"$x(t)$", color = "b")
	ax2.set_ylabel(r"$y(t)$", color = "g")
	ax1.tick_params(axis = "y", labelcolor = "b")
	ax2.tick_params(axis = "y", labelcolor = "g")
	# fig.suptitle(optimizer_method + " - Model " + str(model_id+1), fontsize = 16, y = 0.99)
	# fig.show()
	plt.savefig(os.path.join("output", "model" + str(model_id+1) + "_ic0.png"), bbox_inches = 'tight')
	plt.close()

	# Compute SSE
	sse = ms.compute_SSE(X_test.reshape(simulation.shape), simulation)

	# Set mean SSE to the model
	ms.set_model_SSE(model_id, sse)

# Compute AIC and AICc
best_AIC_model = ms.compute_AIC()
best_AICc_model = ms.compute_AICc()
best_BIC_model = ms.compute_BIC()

# Get best model
print("Melhor modelo AIC = " + str(best_AIC_model+1) + "\n")
print("Melhor modelo AICc = " + str(best_AICc_model+1) + "\n")
print("Melhor modelo BIC = " + str(best_BIC_model+1) + "\n")

# Write results
ms.write_output()
ms.write_AICc_weights()
ms.write_pareto_curve(optimizer_method)
