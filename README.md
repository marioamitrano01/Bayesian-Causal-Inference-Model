# Bayesian Causal Inference Model

This repository contains a Python implementation of a Bayesian causal inference model using a hierarchical framework. The code simulates data, constructs a Bayesian model with PyMC, runs MCMC sampling, and generates interactive plots with Plotly to visualize posterior distributions, trace plots, and posterior predictive checks.

## Features

- **Data Simulation:**  
  Generates synthetic data based on a hierarchical model where the outcome is influenced by treatment assignment and a confounding variable.

- **Hierarchical Bayesian Modeling:**  
  Constructs a Bayesian model with non-informative (weakly informative) priors for group-specific intercepts, treatment effect, confounder coefficient, and error variance.

- **MCMC Inference:**  
  Uses PyMC's NUTS sampler to perform inference and produces diagnostic summaries using ArviZ.

- **Interactive Visualizations:**  
  Creates interactive plots using Plotly for:
  - Posterior distributions of parameters.
  - MCMC trace plots for convergence diagnostics.
  - Posterior predictive checks comparing observed data with model predictions.

## Prerequisites

Ensure you have Python 3.9 or later installed. The following Python libraries are required:

- [numpy](https://numpy.org/)
- [pymc](https://www.pymc.io/)
- [arviz](https://arviz-devs.github.io/arviz/)
- [plotly](https://plotly.com/python/)

You can install these dependencies using pip:

```bash
pip install numpy pymc arviz plotly
