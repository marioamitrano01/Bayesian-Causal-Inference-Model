import numpy as np
import pymc as pm
import arviz as az
import plotly.graph_objects as go
import plotly.express as px
import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper


class BayesianCausalInference:
    def __init__(self, N, p_trt, true_alpha_control, true_alpha_treatment,
                 true_beta, true_gamma, sigma, seed):
        """
        Initialize the model with the following parameters:
        
        Parameters:
          - N: Total number of observations.
          - p_trt: Probability of assignment to the treatment group.
          - true_alpha_control: True intercept for the control group.
          - true_alpha_treatment: True intercept for the treatment group.
          - true_beta: True causal effect of the treatment.
          - true_gamma: True coefficient for the confounding variable.
          - sigma: Standard deviation of the outcome errors.
          - seed: Random seed for reproducibility.
        """
        self.N = N
        self.p_trt = p_trt
        self.true_alpha_control = true_alpha_control
        self.true_alpha_treatment = true_alpha_treatment
        self.true_beta = true_beta
        self.true_gamma = true_gamma
        self.sigma = sigma
        self.seed = seed

        # Placeholders for simulated data
        self.trt = None       # Treatment indicator (0 for control, 1 for treatment)
        self.conf = None      # Confounding variable
        self.y = None         # Observed outcome

        # Results from inference
        self.trace = None     # MCMC trace (ArviZ InferenceData)
        self.model = None     # PyMC model

        np.random.seed(self.seed)

    @timer
    def simulate_data(self):
        """
        Simulate data based on the generative model:
        
            1. Treatment assignment:
               trt_i ~ Bernoulli(p_trt)
               
            2. Confounding variable:
               conf_i ~ N(0, 1)
               
            3. Group-specific intercept:
               α_i = true_alpha_control if trt_i = 0
                     true_alpha_treatment if trt_i = 1
               
            4. Outcome:
               y_i ~ N(α_i + true_beta * trt_i + true_gamma * conf_i, σ²)
        """
        # Generate treatment assignment (0 or 1)
        self.trt = np.random.binomial(1, self.p_trt, size=self.N)
        # Generate confounding variable
        self.conf = np.random.normal(0, 1, size=self.N)
        # Assign intercept based on treatment group
        alpha = np.where(self.trt == 1, self.true_alpha_treatment, self.true_alpha_control)
        # Compute the mean outcome for each individual
        mu = alpha + self.true_beta * self.trt + self.true_gamma * self.conf
        # Generate observed outcome
        self.y = np.random.normal(mu, self.sigma)
        print("Data simulation complete.")

    @timer
    def build_model(self):
        """
        Construct the hierarchical Bayesian model using PyMC.
        
        Model details:
          - Priors for group intercepts:
              α_control, α_treatment ~ N(0, 10²)
          - Prior for treatment effect:
              β ~ N(0, 10²)
          - Prior for confounder coefficient:
              γ ~ N(0, 10²)
          - Prior for error standard deviation:
              σ ~ HalfCauchy(5)
          - Likelihood:
              y_i ~ N(α_i + β * trt_i + γ * conf_i, σ²)
              with α_i determined by treatment group.
        """
        with pm.Model() as self.model:
            # Priors for group intercepts
            alpha_control = pm.Normal("alpha_control", mu=0, sigma=10)
            alpha_treatment = pm.Normal("alpha_treatment", mu=0, sigma=10)
            
            # Prior for treatment effect (causal effect)
            beta = pm.Normal("beta", mu=0, sigma=10)
            
            # Prior for the confounder coefficient
            gamma = pm.Normal("gamma", mu=0, sigma=10)
            
            # Prior for the error standard deviation
            sigma = pm.HalfCauchy("sigma", beta=5)
            
            # Define group-specific intercept based on treatment assignment
            alpha = pm.math.switch(pm.math.eq(self.trt, 1), alpha_treatment, alpha_control)
            
            # Expected outcome given the predictors
            mu_val = alpha + beta * self.trt + gamma * self.conf
            
            # Likelihood function for the observed data
            y_obs = pm.Normal("y_obs", mu=mu_val, sigma=sigma, observed=self.y)
        print("Model built successfully.")

    @timer
    def run_inference(self, draws, tune, chains, target_accept):
        """
        Run MCMC sampling to obtain the posterior distributions of the parameters.
        
        Parameters:
          - draws: Number of samples to draw after tuning.
          - tune: Number of tuning (adaptation) steps.
          - chains: Number of MCMC chains.
          - target_accept: Target acceptance rate for the NUTS sampler.
        
        A diagnostic summary is printed using ArviZ.
        """
        if self.model is None:
            raise ValueError("You must first build the model by calling build_model().")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=self.seed
            )
        print("Inference completed.")
        # Diagnostic summary using ArviZ
        print(az.summary(self.trace, var_names=["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]))

    def plot_posteriors(self):
        """
        Generate interactive plots for the posterior distributions of key parameters:
          - Group intercepts (alpha_control, alpha_treatment)
          - Causal effect (beta)
          - Confounder coefficient (gamma)
          - Error standard deviation (sigma)
          
        Uses Plotly to create interactive histograms with kernel density estimates.
        """
        if self.trace is None:
            raise ValueError("Run inference first with run_inference().")
        
        param_names = ["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]
        fig = go.Figure()
        for param in param_names:
            # Extract samples from the InferenceData object (flatten chains)
            samples = self.trace.posterior[param].values.flatten()
            hist_data = np.histogram(samples, bins=50, density=True)
            bin_centers = 0.5 * (hist_data[1][1:] + hist_data[1][:-1])
            fig.add_trace(go.Scatter(x=bin_centers, y=hist_data[0],
                                     mode='lines',
                                     name=param))
        
        fig.update_layout(title="Posterior Distributions of Parameters",
                          xaxis_title="Parameter Value",
                          yaxis_title="Density",
                          template="plotly_white")
        fig.show()

    def plot_trace(self):
        """
        Generate interactive trace plots for each parameter to assess MCMC convergence.
        """
        if self.trace is None:
            raise ValueError("Run inference first with run_inference().")
        
        param_names = ["alpha_control", "alpha_treatment", "beta", "gamma", "sigma"]
        fig = go.Figure()
        # Loop through each parameter and chain for trace visualization
        for param in param_names:
            chains = self.trace.posterior[param].chain.values
            for ch in chains:
                samples = self.trace.posterior[param].sel(chain=ch).values.flatten()
                fig.add_trace(go.Scatter(y=samples,
                                         mode='lines',
                                         name=f"{param} - chain {ch}"))
        fig.update_layout(title="MCMC Trace Plots",
                          yaxis_title="Parameter Value",
                          template="plotly_white")
        fig.show()

    def posterior_predictive_check(self, num_pp_samples):
        """
        Conduct a Posterior Predictive Check (PPC) by generating predictive samples 
        from the model and comparing them to the observed data.
        
        Parameters:
          - num_pp_samples: Number of posterior predictive samples per chain to use.
        
        In this updated implementation, we manually subsample the posterior trace 
        to include only the first num_pp_samples draws per chain and then call the 
        predictive sampler with return_inferencedata=False.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Ensure that you have built the model and run inference before calling this method.")
        
        # Subsample the posterior trace
        trace_subset = self.trace.copy()
        trace_subset.posterior = trace_subset.posterior.sel(draw=slice(0, num_pp_samples))
        
        with self.model:
            ppc = pm.sample_posterior_predictive(
                trace_subset,
                var_names=["y_obs"],
                random_seed=self.seed,
                return_inferencedata=False  # Return as dictionary
            )
        
        # Extract the predicted y values
        y_ppc = ppc["y_obs"].flatten()
        
        # Create interactive histograms to compare observed and predicted data.
        # To ensure both are visible, we adjust the opacity and add observed data last (so it appears on top).
        fig = go.Figure()
        # Add the posterior predictions first.
        fig.add_trace(go.Histogram(
            x=y_ppc,
            nbinsx=50,
            opacity=0.5,
            name="Posterior Predictions",
            marker_color='red',
            marker_line_color='black',
            marker_line_width=1.0
        ))
        # Add the observed data second.
        fig.add_trace(go.Histogram(
            x=self.y,
            nbinsx=50,
            opacity=0.5,
            name="Observed Data",
            marker_color='blue',
            marker_line_color='black',
            marker_line_width=1.0
        ))
        fig.update_layout(barmode='overlay',
                          title="Posterior Predictive Check",
                          xaxis_title="y value",
                          yaxis_title="Count",
                          template="plotly_white")
        fig.show()


if __name__ == "__main__":
    print("Welcome to the Mario's Bayesian Causal Inference Model.")
    print("Please follow the instructions to input the required parameters.\n")
    
    # User inputs with guidance
    try:
        N = int(input("Enter total number of observations (e.g., 1000): "))
        p_trt = float(input("Enter probability of treatment assignment (0 to 1, e.g., 0.5): "))
        true_alpha_control = float(input("Enter true intercept for control group (e.g., 2.0): "))
        true_alpha_treatment = float(input("Enter true intercept for treatment group (e.g., 2.0): "))
        true_beta = float(input("Enter true causal effect of the treatment (e.g., 3.0): "))
        true_gamma = float(input("Enter true coefficient for the confounding variable (e.g., 1.5): "))
        sigma = float(input("Enter the standard deviation of the outcome errors (e.g., 1.0): "))
        seed = int(input("Enter a random seed (e.g., 42): "))
        
        # MCMC sampling parameters:
        draws = int(input("Enter number of MCMC draws (post-tuning, e.g., 2000): "))
        tune = int(input("Enter number of tuning steps (e.g., 1000): "))
        chains = int(input("Enter number of MCMC chains (e.g., 4): "))
        target_accept = float(input("Enter target acceptance rate for NUTS sampler (e.g., 0.9): "))
        num_pp_samples = int(input("Enter number of posterior predictive samples per chain (e.g., 200): "))
    except Exception as e:
        print("Invalid input. Please restart and enter valid numerical values.")
        raise e

    # Initialize the Bayesian causal inference model
    model_instance = BayesianCausalInference(
        N=N,
        p_trt=p_trt,
        true_alpha_control=true_alpha_control,
        true_alpha_treatment=true_alpha_treatment,
        true_beta=true_beta,
        true_gamma=true_gamma,
        sigma=sigma,
        seed=seed
    )
    
    # Simulate data
    model_instance.simulate_data()
    
    # Build the Bayesian model
    model_instance.build_model()
    
    # Run MCMC inference
    model_instance.run_inference(draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    
    # Plot posterior distributions
    model_instance.plot_posteriors()
    
    # Plot trace plots for MCMC diagnostics
    model_instance.plot_trace()
    
    # Conduct posterior predictive check with subsampled trace
    model_instance.posterior_predictive_check(num_pp_samples=num_pp_samples)
    
    print("Bayesian causal inference complete. Explore the interactive plots for detailed diagnostics.")
