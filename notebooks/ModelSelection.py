#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern, RBF, WhiteKernel, RationalQuadratic, DotProduct, ConstantKernel


# In[2]:


# The following models are phenomenological, based on Banks et al. (1987)

def model1(X, t):
	x, y = X
	dXdt = [3.705409 * x - 0.875407 * x * y / ( 1 + 2.0925471112633475e-16 * x  ),
		1.136291807280769e-05 * x - 0.0007703183454985423 * y * y
	]
	return dXdt

model1_vars = {
	"k": 5,
	"X0": np.array([2400.5117688174696, 4.1470588235293935])
}

def model2(X, t):
	x, y = X
	dXdt = [4.699038 * x - 1.115438 * x * y / ( 1 + 2.5834261183184556e-17 * x  ),
		8.917509704299429e-06 * x + 3.706581216964794e-26 * y - 0.0006152529575199094 * y * y
	]
	return dXdt

model2_vars = {
	"k": 6,
	"X0": np.array([2400.5117688174696, 4.1470588235293935])
}

# The following models came from the SINDy-SA

def model3(X, t): # in the previous this was model 02
	x, y = X
	dXdt = [-1.002712e-04*x**2 + -0.007341637777931233*x*y + 4.070982e-08*x**3,
		0.000e+00
	]
	return dXdt

model3_vars = {
	"k": 3,
	"X0": np.array([2713.7232754553243, 4.147055452699332])
}

def model4(X, t): # in the previous this was model 09
	x, y = X
	dXdt = [-1.246743e-04*x**2 + 5.017520e-08*x**3 + -0.0016170094762767212*x*y**2,
		0.000e+00
	]
	return dXdt

model4_vars = {
	"k": 3,
	"X0": np.array([2681.4743756420753, 4.1470600152530155])
}

def model5(X, t):  # in the previous this was model 14
	x, y = X
	dXdt = [-7.713189428726371e-06*x**2 + 5.340231e-08*x**3 + -3.281022e-05*x**2*y + -0.0002904897329707499*x*y**2,
		0.000e+00
	]
	return dXdt

model5_vars = {
	"k": 4,
	"X0": np.array([2719.3794153268414, 4.147056509871602])
}


# In[3]:


def compute_SSE(target, predicted):
	squared_errors = (target - predicted)**2.0
	return np.sum(squared_errors)


# In[4]:


def compute_AIC(n, k, SSE):
	AIC = n*np.log(SSE/n) + 2.0*k
	AICmin = np.amin(AIC)
	Delta_AIC = AIC - AICmin
	like = np.exp(-0.5*Delta_AIC)
	likesum = np.sum(like)
	AIC_weights = like/likesum
	print("Melhores modelos AIC = " + str(np.argsort(AIC_weights)+1) + "\n")
	best_AIC_model = np.argmax(AIC_weights)
	AIC_evid_ratio = AIC_weights[best_AIC_model]/AIC_weights
	return best_AIC_model


# In[5]:


def compute_AICc(n, k, SSE):
	AICc = n*np.log(SSE/n) + 2.0*k  + (2.0*k*(k + 1.0))/(n - k - 1.0)
	AICcmin = np.amin(AICc)
	Delta_AICc = AICc - AICcmin
	likec = np.exp(-0.5*Delta_AICc)
	likecsum = np.sum(likec)
	AICc_weights = likec/likecsum
	print("Melhores modelos AICc = " + str(np.argsort(AICc_weights)+1) + "\n")
	best_AICc_model = np.argmax(AICc_weights)
	AICc_evid_ratio = AICc_weights[best_AICc_model]/AICc_weights
	return best_AICc_model


# In[6]:


def compute_BIC(n, k, SSE):
	BIC = n*np.log(SSE/n) + k*np.log(n)
	BICmin = np.amin(BIC)
	Delta_BIC = BIC - BICmin
	BICsum = np.sum(np.exp(-0.5*Delta_BIC))
	BIC_prob = np.exp(-0.5*Delta_BIC)/BICsum
	print("Melhores modelos BIC = " + str(np.argsort(BIC_prob)+1) + "\n")
	best_BIC_model = np.argmax(BIC_prob)
	return best_BIC_model


# In[7]:


# Read train data
data = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t = data["Time"]
X_aphid = data["Aphid"].reshape((-1, 1))
X_ladybeetle = data["Ladybeetle"].reshape((-1, 1))
X = np.hstack((X_aphid, X_ladybeetle))

X0 = X[0, :]
t_steps = len(t)


# In[8]:


# Read test data
data_test = np.genfromtxt("data.csv", dtype = float, delimiter = ',', names = True)
t_test = data_test["Time"]
X_test_aphid = data_test["Aphid"].reshape((-1, 1))
X_test_ladybeetle = data_test["Ladybeetle"].reshape((-1, 1))
X_test = np.hstack((X_test_aphid, X_test_ladybeetle))

X0_test = X_test[0, :]


# In[9]:


# Simulate models
model1_vars["simulation"] = odeint(model1, model1_vars["X0"], t_test)
model2_vars["simulation"] = odeint(model2, model2_vars["X0"], t_test)
model3_vars["simulation"] = odeint(model3, model3_vars["X0"], t_test)
model4_vars["simulation"] = odeint(model4, model4_vars["X0"], t_test)
model5_vars["simulation"] = odeint(model5, model5_vars["X0"], t_test)


# In[10]:


# Compute SSE
model1_vars["SSE"] = compute_SSE(X_test.reshape(model1_vars["simulation"].shape), model1_vars["simulation"])
model2_vars["SSE"] = compute_SSE(X_test.reshape(model2_vars["simulation"].shape), model2_vars["simulation"])
model3_vars["SSE"] = compute_SSE(X_test.reshape(model3_vars["simulation"].shape), model3_vars["simulation"])
model4_vars["SSE"] = compute_SSE(X_test.reshape(model4_vars["simulation"].shape), model4_vars["simulation"])
model5_vars["SSE"] = compute_SSE(X_test.reshape(model5_vars["simulation"].shape), model5_vars["simulation"])


# In[11]:


# Define information criteria parameters
k = np.array([model1_vars["k"],
	model2_vars["k"],
	model3_vars["k"],
	model4_vars["k"],
	model5_vars["k"],
])
SSE = np.array([model1_vars["SSE"],
	model2_vars["SSE"],
	model3_vars["SSE"],
	model4_vars["SSE"],
	model5_vars["SSE"],
])


# In[12]:


# Compute AIC and AICc
best_AIC_model = compute_AIC(t_steps, k, SSE)
best_AICc_model = compute_AICc(t_steps, k, SSE)
best_BIC_model = compute_BIC(t_steps, k, SSE)

