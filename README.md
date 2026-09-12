# SMKLE: superseded by the skmle package

This repository holds the original R simulation code for the paper below. It is
kept so the published simulation study stays reproducible, and is no longer
developed.

**For applied use, install [skmle](https://github.com/dayusun/skmle)**, an R
package implementing the same estimator with a C++ backend via Rcpp, automatic
bandwidth selection, ordinary `coef()` / `vcov()` / `summary()` methods, and
vignettes. Documentation: https://www.sundayu.me/skmle/

## Method

The **Sieve Maximum Kernel-weighted Log-likelihood Estimator (SMKLE)** fits
transformed hazards models when a time-dependent covariate is observed only
intermittently, at times unrelated to the event time: sparse, asynchronous
longitudinal measurements against a censored survival outcome. Rather than
carrying the last value forward or smoothing the covariate and substituting it,
SMKLE weights each covariate measurement by how far its observation time sits
from the time being modelled, and estimates the infinite-dimensional nuisance
parameter by sieves.

## Paper

Sun, D., Sun, Z., Zhao, X., and Cao, H. (2025). Kernel Meets Sieve: Transformed
Hazards Models with Sparse Longitudinal Covariates. *Journal of the American
Statistical Association*, **120**(552), 2580-2591.
https://doi.org/10.1080/01621459.2025.2476781

Preprint: https://arxiv.org/abs/2308.15549

## Contents

R scripts reproducing the simulation study.

| File | Purpose |
|------|---------|
| `multiple_4_fixed_cons.R` | SMKLE at a fixed bandwidth |
| `multiple_4_cv_cons.R` | SMKLE with cross-validated bandwidth |
| `multiple_6_LVCF.R` | Last-value-carried-forward comparator |
| `multiple_Cao_Cox_fixed.R` | Cao et al. kernel estimating equations, proportional hazards, fixed bandwidth |
| `multiple_Cao_Cox_CV.R` | As above, cross-validated bandwidth |
| `multiple_Cao_additive_fixed.R` | Cao et al. kernel estimating equations, additive hazards, fixed bandwidth |
| `multiple_Cao_additive_CV.R` | As above, cross-validated bandwidth |

## Author

Dayu Sun, Department of Biostatistics and Health Data Science, Indiana
University School of Medicine. https://www.sundayu.me/
