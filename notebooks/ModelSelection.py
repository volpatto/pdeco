# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct, ConstantKernel


# +
# The following models are phenomenological, based on Banks et al. (1987)

def model1(X, t):
	x, y = X
	dXdt = [0.318752 * x - 0.080779 * x * y / ( 1 + 0.0001948159776916025 * x  ),
		0.0002897334690674872 * x - 0.012509 * y * y
	]
	return dXdt

model1_vars = {
	"k": 5,
	"X0": np.array([2400.5117688174696, 4.1470588235293935])
}

def model2(X, t):
	x, y = X
	dXdt = [0.295637 * x - 0.083731 * x * y / ( 1 + 0.00019604301622985423 * x ),
		7.673479420465873e-05 * x * y / ( 1 + 0.00019604301622985423 * x ) - 0.016822 * y * y
	]
	return dXdt

model2_vars = {
	"k": 5,
	"X0": np.array([2400.5117688174696, 4.1470588235293935])
}

def model3(X, t):
	x, y = X
	dXdt = [0.25 * x - 0.000297 * x * x * y / ( 1 + 0.000003 * x * x ),
		1.668734524839995e-06 * x * x * y / ( 1 + 0.000003 * x * x ) - 0.102103 * y * y
	]
	return dXdt

model3_vars = {
	"k": 5,
	"X0": np.array([2400.5117688174696, 4.1470588235293935])
}

# The following models came from the SINDy-SA

def model4(X, t):# in the previous this was model 14
	x, y = X
	dXdt = [-7.712e-06*x**2 + 5.228383e-08*x**3 + -3.212072e-05*x**2*y + -0.0002902*x*y**2,
		0.000e+00
	]
	return dXdt

model4_vars = {
	"k": 4,
	"X0": np.array([2719.3794153268414, 4.147056509871602])
}

def model5(X, t):# in the previous this was model 02
	x, y = X
	dXdt = [-9.722941e-05*x**2 + -0.007269*x*y + 3.944668e-08*x**3,
		0.000e+00
	]
	return dXdt

model5_vars = {
	"k": 3,
	"X0": np.array([2713.7232754553243, 4.147055452699332])
}

def model6(X, t):# in the previous this was model 09
	x, y = X
	dXdt = [-9.161618e-05*x**2 + 3.746377e-08*x**3 + -0.001601*x*y**2,
		0.000e+00
	]
	return dXdt

model6_vars = {
	"k": 3,
	"X0": np.array([2681.4743756420753, 4.1470600152530155])
}


# -

def compute_SSE(target, predicted):
	squared_errors = (target - predicted)**2.0
	return np.sum(squared_errors)


def compute_AIC(n, k, SSE):
	AIC = n*np.log(SSE/n) + 2.0*k
	AICmin = np.amin(AIC)
	Delta_AIC = AIC - AICmin
	like = np.exp(-0.5*Delta_AIC)
	likesum = np.sum(like)
	AIC_weights = like/likesum
	print("Melhores modelos AIC (worst to best) = " + str(np.argsort(AIC_weights)+1) + "\n")
	print("Melhores modelos AIC (worst to best) = " + str(np.sort(AIC_weights)) + "\n")
	best_AIC_model = np.argmax(AIC_weights)
	AIC_evid_ratio = AIC_weights[best_AIC_model]/AIC_weights
	return best_AIC_model


def compute_AICc(n, k, SSE):
	AICc = n*np.log(SSE/n) + 2.0*k  + (2.0*k*(k + 1.0))/(n - k - 1.0)
	AICcmin = np.amin(AICc)
	Delta_AICc = AICc - AICcmin
	likec = np.exp(-0.5*Delta_AICc)
	likecsum = np.sum(likec)
	AICc_weights = likec/likecsum
	print("Melhores modelos AICc (worst to best) = " + str(np.argsort(AICc_weights)+1) + "\n")
	print("Melhores modelos AICc (worst to best) = " + str(np.sort(AICc_weights)) + "\n")
	best_AICc_model = np.argmax(AICc_weights)
	AICc_evid_ratio = AICc_weights[best_AICc_model]/AICc_weights
	return best_AICc_model


def compute_BIC(n, k, SSE):
	BIC = n*np.log(SSE/n) + k*np.log(n)
	BICmin = np.amin(BIC)
	Delta_BIC = BIC - BICmin
	BICsum = np.sum(np.exp(-0.5*Delta_BIC))
	BIC_prob = np.exp(-0.5*Delta_BIC)/BICsum
	print("Melhores modelos BIC (worst to best) = " + str(np.argsort(BIC_prob)+1) + "\n")
	print("Melhores modelos BIC (worst to best) = " + str(np.sort(BIC_prob)) + "\n")
	best_BIC_model = np.argmax(BIC_prob)
	return best_BIC_model

# +
# Read train data
data_dir = "../data/"
data = np.genfromtxt(data_dir + "data.csv", dtype = float, delimiter = ',', names = True)
t = data["Time"]
X_aphid = data["Aphid"].reshape((-1, 1))
X_ladybeetle = data["Ladybeetle"].reshape((-1, 1))
X = np.hstack((X_aphid, X_ladybeetle))

X0 = X[0, :]
t_steps = len(t)

# +
# Read test data
data_dir = "../data/"
data_test = np.genfromtxt(data_dir + "data.csv", dtype = float, delimiter = ',', names = True)
t_test = data_test["Time"]
X_test_aphid = data_test["Aphid"].reshape((-1, 1))
X_test_ladybeetle = data_test["Ladybeetle"].reshape((-1, 1))
X_test = np.hstack((X_test_aphid, X_test_ladybeetle))

X0_test = X_test[0, :]
# -

# Simulate models
model1_vars["simulation"] = odeint(model1, model1_vars["X0"], t_test)
model2_vars["simulation"] = odeint(model2, model2_vars["X0"], t_test)
model3_vars["simulation"] = odeint(model3, model3_vars["X0"], t_test)
model4_vars["simulation"] = odeint(model4, model4_vars["X0"], t_test)
model5_vars["simulation"] = odeint(model5, model5_vars["X0"], t_test)
model6_vars["simulation"] = odeint(model6, model6_vars["X0"], t_test)

# Compute SSE
model1_vars["SSE"] = compute_SSE(X_test.reshape(model1_vars["simulation"].shape), model1_vars["simulation"])
model2_vars["SSE"] = compute_SSE(X_test.reshape(model2_vars["simulation"].shape), model2_vars["simulation"])
model3_vars["SSE"] = compute_SSE(X_test.reshape(model3_vars["simulation"].shape), model3_vars["simulation"])
model4_vars["SSE"] = compute_SSE(X_test.reshape(model4_vars["simulation"].shape), model4_vars["simulation"])
model5_vars["SSE"] = compute_SSE(X_test.reshape(model5_vars["simulation"].shape), model5_vars["simulation"])
model6_vars["SSE"] = compute_SSE(X_test.reshape(model6_vars["simulation"].shape), model6_vars["simulation"])

# Define information criteria parameters
k = np.array([model1_vars["k"],
	model2_vars["k"],
	model3_vars["k"],
	model4_vars["k"],
	model5_vars["k"],
	model6_vars["k"],
])
SSE = np.array([model1_vars["SSE"],
	model2_vars["SSE"],
	model3_vars["SSE"],
	model4_vars["SSE"],
	model5_vars["SSE"],
	model6_vars["SSE"],
])
print("SSE = " + str(SSE) + "\n")

# Compute AIC, AICc, and BIC
best_AIC_model = compute_AIC(t_steps, k, SSE)
best_AICc_model = compute_AICc(t_steps, k, SSE)
best_BIC_model = compute_BIC(t_steps, k, SSE)
