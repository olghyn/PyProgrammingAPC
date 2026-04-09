#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:43:27 2026

@author: olga
"""
import sys
sys.path.append("./")
from my_MH import MetropolisHastingsSampler
import numpy as np
import matplotlib.pyplot as plt
# High-resolution plots for scripts
plt.rcParams["figure.dpi"] = 300


rng = np.random.default_rng(seed=42)

# Choose the "true" parameters.

m_true = -0.9594
b_true = 4.294
f_true = 0.534

# Generate some synthetic data from the model.

N_data = 50

xobs = np.sort(10 * rng.normal(size=N_data))
yerr = 0.1 + .5 * rng.normal(size=N_data)
yobs = m_true * xobs + b_true
yobs += np.abs(f_true * yobs) * rng.standard_normal(size=N_data)
yobs += yerr * rng.standard_normal(size=N_data)
yerr = np.abs(yerr)

fig, ax = plt.subplots(1,1)
_ = ax.errorbar(xobs, yobs, yerr=yerr, fmt=".k", capsize=0, label="data")
x0 = np.linspace(xobs.min(), xobs.max(), 500)
_ = ax.plot(x0, m_true * x0 + b_true, "k", alpha=0.3, lw=3, label='underlying model')
_ = ax.set(xlabel='x', ylabel='y')
plt.legend()
plt.show()

# Guess some values as priors

prior = [
    [-2.0,0.0],
    [2.0,6.0],
    [-4.0, 4.0]
]

# Define log prior and log likelihood probabilities, compute log posterior

def log_prior(theta, prior):
    m, b, log_f = theta
    mlim, blim, flim = prior
    if mlim[0] < m < mlim[1] and blim[0] < b < blim[1] and flim[0] < log_f < flim[1]:
        return 0.0
    return -np.inf

def log_likelihood(theta, xx, yy, ee):
    m, b, log_f = theta
    model = m * xx + b
    sigma2 = ee**2 + model**2 * np.exp(2 * log_f)
    log_l = -0.5 * np.sum((yy - model) ** 2 / sigma2 + np.log(sigma2*2*np.pi))
    return log_l

def log_probability(theta, xx, yy, ee, prior):
    lp = log_prior(theta, prior)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, xx, yy, ee)

# Run my Metropolis Hastings algoritm

scales = [0.1, 0.5, 0.01]
nsteps = 100000
mh = MetropolisHastingsSampler(log_probability, prior, xobs, yobs, yerr, scales, nsteps)
mh.run(use_multivariate=False)

# Print some useful values

print("Acceptance rate:", mh.get_acceptance_rate())
print("Median parameters:", mh.get_median_params())
print("Autocorrelation times:", mh.get_autocorr_times())
print("Estimated thinning interval:", mh.estimate_thin())
print("Burn-in steps:", mh.burn)
print("Estimated covariance matrix:\n", mh.compute_covariance())

# Plot results to check convergence
mh.plot_traces()
mh.plot_corner(truths=[m_true, b_true, np.log(f_true)])
mh.plot_convergence(labels=["m", "b", "logf"])

clean_chain = (nsteps-mh.burn)/mh.estimate_thin()
print("after thinning and burning chain would be: ", clean_chain)

N_eff = clean_chain/(2*mh.get_autocorr_times())

bad = N_eff < 100
if bad.any():
    print("fit didn't converge for parameters:", np.where(bad)[0])
else: print("fit has converged!")
#%%
# Run again but this time use a multivariate gaussian approach to keep track of the correlation between m and b
scales = [0.1, 0.5, 0.01]
nsteps = 100000
mh = MetropolisHastingsSampler(log_probability, prior, xobs, yobs, yerr, scales, nsteps)
mh.run(use_multivariate=True)

# Print some useful values

print("Acceptance rate:", mh.get_acceptance_rate())
print("Median parameters:", mh.get_median_params())
print("Autocorrelation times:", mh.get_autocorr_times())
print("Estimated thinning interval:", mh.estimate_thin())
print("Burn-in steps:", mh.burn)
print("Estimated covariance matrix:\n", mh.compute_covariance())

# Plot results to check convergence
mh.plot_traces()
mh.plot_corner(truths=[m_true, b_true, np.log(f_true)])
mh.plot_convergence(labels=["m", "b", "logf"])

clean_chain = (nsteps-mh.burn)/mh.estimate_thin()

N_eff = clean_chain/(2*mh.get_autocorr_times())
bad = N_eff < 100
if bad.any():
    print("fit didn't converge for parameters:", np.where(bad)[0])
else: print("fit has converged!")