#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:02:06 2026

@author: olga
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import itertools


class MetropolisHastingsSampler:
    def __init__(self, log_prob_fn, prior, xobs, yobs, yerr, scales, nsteps, burn_frac=0.2, thin=10, seed=0):
        
        """
        Metropolis-Hastings MCMC sampler for a single walker.
        
        This class implements a basic Metropolis-Hastings algorithm to sample
        from the posterior distribution of model parameters given observed data.
        It supports automatic tuning of proposal step sizes during burn-in, 
        computation of autocorrelation times, thinning, and plotting diagnostics.
    
        Parameters
        ----------
        log_prob_fn : function
            Function that returns the log-posterior probability of parameters.
            Signature: log_prob_fn(theta, xobs, yobs, yerr, prior)
        prior : list of lists
            List of parameter bounds: [[m_min,m_max], [b_min,b_max], [logf_min,logf_max]]
        xobs : array_like
            Observed x-values.
        yobs : array_like
            Observed y-values.
        yerr : array_like
            Observational uncertainties.
        scales : array_like
            Initial step sizes (standard deviations) for each parameter.
        nsteps : int
            Total number of MCMC steps.
        burn_frac : float, optional
            Fraction of steps to discard as burn-in (default 0.2).
        thin : int, optional
            Initial thinning interval (default 10). Can be updated automatically.
        seed : int, optional
            Random seed for reproducibility (default 0).
    
        Attributes
        ----------
        theta_chain : list
            List storing sampled parameter vectors.
        post : list
            List storing log-posterior probabilities for the chain.
        acc_count : int
            Number of accepted steps.
        burn : int
            Number of steps considered burn-in.
        rng : numpy.random.Generator
            Random number generator instance.
        scales : ndarray
            Current step sizes for proposal distribution.
        
        Methods
        -------
        initialize_theta()
            Initialize the chain with a random sample from the prior.
        run()
            Run the Metropolis-Hastings MCMC sampler.
        get_chain()
            Return the chain after burn-in and thinning.
        get_acceptance_rate()
            Return the overall acceptance rate of the sampler.
        get_median_params()
            Compute median parameter values from the thinned chain.
        get_autocorr_times()
            Compute autocorrelation times for each parameter.
        estimate_thin()
            Estimate an appropriate thinning interval from autocorrelation times.
        plot_corner(truths=None)
            Produce a corner plot of the thinned chain.
        plot_traces(labels=None)
            Produce trace plots for each parameter.
        plot_convergence(labels=None)
            Plot estimated autocorrelation time vs chain length to check convergence.
    
        Notes
        -----
        - Automatic tuning of proposal scales occurs every 100 steps during burn-in.
        - Autocorrelation times can be used to guide thinning of the chain.
        - Convergence plots can help visualize whether the chain length is sufficient.
        """
        
        self.log_prob_fn = log_prob_fn
        self.prior = prior
        self.xobs = xobs
        self.yobs = yobs
        self.yerr = yerr
        self.scales = np.array(scales)
        self.nsteps = nsteps
        self.burn = int(nsteps * burn_frac)
        self.thin = thin
        self.rng = np.random.default_rng(seed)

        # chain storage
        self.theta_chain = []
        self.post = []
        self.acc_count = 0

    def initialize_theta(self):
        
        """
        Initialize the Markov chain.
    
        Draws an initial parameter vector uniformly from the prior bounds
        and evaluates its log-posterior probability. Stores both in the chain.
        """
        
        theta0 = [self.rng.uniform(low=self.prior[i][0], high=self.prior[i][1]) for i in range(3)]
        p0 = self.log_prob_fn(theta0, self.xobs, self.yobs, self.yerr, self.prior)
        self.theta_chain.append(theta0)
        self.post.append(p0)

    def run(self, use_multivariate=False):
        
        """
        Run the Metropolis-Hastings MCMC sampler.
        
        Performs `nsteps` iterations of the algorithm:
        - Proposes new parameters using a Gaussian centered on the current state.
        - Accepts or rejects proposals using the Metropolis criterion: 
            a random number r between 0 and 1 is extracted and new draws are accepted if
            ratio(p_i+1/p_i) > r.
        - Stores the resulting chain and log-posterior values.
        
        During burn-in:
        - Automatically tunes proposal step sizes every 100 steps
          to keep the acceptance rate between ~0.2 and ~0.5.
        
        Updates
        -------
        theta_chain : list
            Filled with sampled parameter values.
        post : list
            Filled with corresponding log-posterior values.
        acc_count : int
            Total number of accepted proposals.
    
        Parameters
        ----------
        use_multivariate : bool
            If True, use multivariate Gaussian proposal based on covariance.
        """
        self.initialize_theta()
        last_acc_count = 0
    
        cov = None  # will be updated later
    
        for i in range(1, self.nsteps):
            theta_prev = np.array(self.theta_chain[-1])
    
            # --- Proposal ---
            if use_multivariate and cov is not None:
                theta_i = self.rng.multivariate_normal(theta_prev, cov)
            else:
                theta_i = np.array([
                    self.rng.normal(theta_prev[j], self.scales[j])
                    for j in range(3)
                ])
    
            p_i = self.log_prob_fn(theta_i, self.xobs, self.yobs, self.yerr, self.prior)
    
            # --- Acceptance ---
            p_acc = p_i - self.post[-1]
    
            if p_acc >= np.log(self.rng.uniform()):
                self.theta_chain.append(theta_i.tolist())
                self.post.append(p_i)
                self.acc_count += 1
            else:
                self.theta_chain.append(theta_prev.tolist())
                self.post.append(self.post[-1])
    
            # --- Adaptive scaling during burn-in ---
            if i < self.burn and i % 100 == 0:
                local_acc_rate = (self.acc_count - last_acc_count) / 100
                last_acc_count = self.acc_count
    
                if local_acc_rate < 0.2:
                    self.scales *= 0.8
                elif local_acc_rate > 0.5:
                    self.scales *= 1.2
    
            # --- Update covariance AFTER some steps ---
            if use_multivariate and i > self.burn and i % 500 == 0:
                cov = self.compute_covariance()
    
                # Regularization (VERY important!)
                cov += 1e-6 * np.eye(len(cov))

    def get_chain(self):
        
        """
        Return the processed MCMC chain.
    
        Applies burn-in removal and thinning.
    
        Returns
        -------
        chain : ndarray of shape (n_samples, ndim)
            The thinned chain after discarding burn-in samples.
        """
        
        chain = np.array(self.theta_chain)
        return chain[self.burn::self.thin]

    def get_acceptance_rate(self):
        
        """
        Compute the acceptance rate of the sampler.
    
        Returns
        -------
        float
            Fraction of accepted proposals over total steps.
        """
        
        return self.acc_count / self.nsteps

    def get_median_params(self):
        
        """
        Compute median parameter estimates after burning-in.
    
        Returns
        -------
        list
            Median values of each parameter from the processed chain.
        """
        
        chain = self.get_chain()
        return [np.median(chain[:, j]) for j in range(3)]

    # Autocorrelation utilities
    @staticmethod
    def next_pow_two(n):
        i = 1
        while i < n:
            i <<= 1
        return i

    @staticmethod
    def autocorr_func_1d(x):
        
        """
        Compute the 1D autocorrelation function using FFT.
    
        Parameters
        ----------
        x : array_like
    
        Returns
        -------
        acf : ndarray
            Normalized autocorrelation function.
        """
        
        n = MetropolisHastingsSampler.next_pow_two(len(x))
        f = np.fft.fft(x - np.mean(x), n=2*n)
        acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
        acf /= 4 * n
        acf /= acf[0]
        return acf

    @staticmethod
    def auto_window(taus, c=5.0):
        
        """
        Determine the truncation window for autocorrelation time estimation.
    
        Parameters
        ----------
        taus : ndarray
            Cumulative autocorrelation estimates.
        c : float, optional
            Windowing constant (default 5.0).
    
        Returns
        -------
        int
            Index where the window condition fails.
        """
        
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    @staticmethod
    def autocorr_time_mh(x, c=5.0):
        
        """
        Estimate the integrated autocorrelation time.
    
        Parameters
        ----------
        x : array_like
            Chain samples for a single parameter.
        c : float, optional
            Windowing constant.
    
        Returns
        -------
        float
            Estimated autocorrelation time.
        """
            
        f = MetropolisHastingsSampler.autocorr_func_1d(x)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = MetropolisHastingsSampler.auto_window(taus, c)
        return taus[window]

    def get_autocorr_times(self):
        chain = self.get_chain()
        taus = [self.autocorr_time_mh(chain[:, j]) for j in range(chain.shape[1])]
        return np.array(taus)
    
    # Estimate thinning from autocorrelation times
    def estimate_thin(self):
        """
        Compute a good thinning interval based on the autocorrelation times
        of the chain (after burn-in). Returns an integer >= 1.
        """
        chain = np.array(self.theta_chain)[self.burn:]  # skip burn-in
        taus = [self.autocorr_time_mh(chain[:, j]) for j in range(chain.shape[1])]
        thin_est = max(1, int(np.min(taus)))  # at least 1
        return thin_est
    
    def compute_covariance(self):
        """
        Estimate covariance matrix from the chain (after burn-in).
    
        Returns
        -------
        cov : ndarray
            Estimated covariance matrix.
        """
        chain = np.array(self.theta_chain)[self.burn:]
        cov = np.cov(chain, rowvar=False)
        
        return cov

    # Corner plot
    def plot_corner(self, truths=None):
        chain = self.get_chain()
        # use median as default truths if none provided
        par_fit = self.get_median_params()
        
        fig = corner.corner(chain, labels=["m", "b", "logf"], truths=par_fit, show_titles=True, color='dodgerblue')
        if truths is not None:
            truths_array = np.array(truths)
            corner.overplot_lines(fig, truths_array, color="C1", linestyle='dashed')
            corner.overplot_points(fig, truths_array[None, :], marker="s", color="C1")
        plt.show()

    # Trace plot
    def plot_traces(self, labels=None):
        
        """
       Plot trace (time series) of each parameter.
    
       Parameters
       ----------
       labels : list of str, optional
           Names of the parameters. Defaults to ["m", "b", "logf"].
    
       Notes
       -----
       - Shows burn-in region.
       - Displays median value for each parameter.
       """
   
        if labels is None:
            labels = ["m", "b", "logf"]
        chain = np.array(self.theta_chain)
        medians = [np.median(chain[:, j]) for j in range(chain.shape[1])]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        for i in range(3):
            axs[i].scatter(range(len(chain)), chain[:, i], s=1, alpha=0.3, label='Samples')
            axs[i].axvline(self.burn, color='r', linestyle='--', label='burn-in')
            axs[i].axhline(medians[i], color='cyan', linestyle='-.', label=f'Median: {medians[i]:.3f}')
            axs[i].set_ylabel(labels[i])
            axs[i].legend(loc='upper right', markerscale=5)
        plt.xlabel("Iteration step")
        plt.tight_layout()
        plt.show()
        
    def plot_convergence(self, labels=None):
        
        """
        Plot autocorrelation time vs chain length to assess convergence.
    
        For each parameter:
        - Computes τ(N) for increasing chain lengths.
        - Compares it to the heuristic line N/50.
    
        Parameters
        ----------
        labels : list of str, optional
            Names of the parameters.
    
        Notes
        -----
        Convergence is indicated when:
        - τ stabilizes (flattens),
        - and remains well below N/50.
        """
    
        if labels is None:
            labels = ["m", "b", "logf"]
        
        chain = self.get_chain()
        N_steps = np.exp(np.linspace(np.log(100), np.log(len(chain)), 10)).astype(int)
        
        for p in range(chain.shape[1]):
            tau_est = []
            for n in N_steps:
                tau_est.append(self.autocorr_time_mh(chain[:n, p]))
            
            plt.figure()
            plt.loglog(N_steps, tau_est, "o-", label=r"Estimated $\tau$")
            plt.plot(N_steps, N_steps / 50, "--k", label="N/50")
            plt.xlabel("N")
            plt.ylabel(r"$\tau$")
            plt.title(labels[p])
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.show()
            
    def plot_covariance(self, labels=None):
        """
        Plot covariance matrix and parameter correlation scatter.
    
        This diagnostic helps identify correlations between parameters,
        which can strongly affect MCMC efficiency.
    
        Parameters
        ----------
        labels : list of str, optional
            Names of the parameters. Defaults to ["m", "b", "logf"].
    
        Returns
        -------
        cov : ndarray
            Estimated covariance matrix of the chain.
        """
        if labels is None:
            labels = ["m", "b", "logf"]
    
        # Use processed chain (after burn-in and thinning)
        chain = self.get_chain()
    
        

        for i, j in itertools.combinations(range(chain.shape[1]), 2):
            plt.figure()
            plt.scatter(chain[:, i], chain[:, j], s=5, alpha=0.5)
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            plt.title(f"{labels[i]} vs {labels[j]}")
            plt.grid(True)
            plt.show()
            
                    