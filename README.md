# Bayesian Estimation of Decay Parameters in Hawkes Processes
### Description
This code repository replicates the Bayesian approach and the results presented in the paper "Bayesian Estimation of Decay Parameters in Hawkes Processes". 

Each ".py" script (except for "constants.py") addresses a specific section and/or figure of the paper, as summarized below (cf. the comments in the header of each script for more details).
- uncertainty_basics.py exemplifies the remarkable uncertainty properties of decay values (in Hawkes processes with an exponential kernel) as estimated via L-BFGS-B.
- loglik_plot.py visualizes the Hawkes process log-likelihood as a function of the decay.
- uncertainty_influence.py implements, with synthetic data, our Bayesian approach to quantify uncertainty in temporal depencencies inferred via Hawkes processes.
- real_world_earthquakes.py illustrates the estimation of uncertainty in inferred temporal influence with a dataset of earthquakes and aftershocks in Japan.
- misestimation_stationarity.py implements, with synthetic data, our Bayesian approach to address mis-estimation of the decay and estimate the decay in the presence of breaks in Hawkes process stationarity.
- real_world_vocabulary.py illustrates how our Bayesian approach surfaces decay mis-estimation with data from the Duolingo language learning app.
- real_world_effervescence.py applies our Bayesian approach to estimate decay values in a real-world dataset with a stationarity break, namely a dataset of Tweets surrounding a terrorist attack in Paris.

Please note that there may be small quantitative deviations in the exact values inferred via Bayesian inference, as there is inherent randomness in (among others) simulating from Hawkes processes or MCMC convergence, and we do not set random seeds. However, the results produced by these scripts closely follow those of the paper, and in particular the scripts reproduce all results qualitatively.

### Requirements
This code leverages Python v3.7, R v3.5.2, Python packages for Hawkes process modelling and R packages (mainly) for plotting. See requirements.txt for details on the Python packages (or install them using `pip3 install requirements.txt`). Required R packages are [dplyr](https://cran.r-project.org/package=dplyr) v1.0.2, [ggpubr](https://cran.r-project.org/package=ggpubr) v0.4.0, [gtools](https://cran.r-project.org/package=gtools) v3.8.1 and [tidyr](https://cran.r-project.org/package=tidyr) v1.1.2.

### Citation
If you find this research helpful to your work, please consider citing:
```
Citation to be updated
```
