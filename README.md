# Description

The repository contains R codes for the simulation in Sun, D., Sun, Z., Zhao, X., & Cao, H. (2023). Kernel meets sieve: Transformed hazards models with sparse longitudinal covariates. arXiv. http://arxiv.org/abs/2308.15549

The manuscript proposed the **S**ieve **M**aximum **K**ernel-weighed **L**og-likelihood **E**stimation (SMKLE).

> We study the transformed hazards model with time-dependent covariates observed intermittently for the censored outcome. Existing work assumes the availability of the whole trajectory of the time-dependent covariates, which is unrealistic. We propose to combine kernel-weighted log-likelihood and sieve maximum log-likelihood estimation to conduct statistical inference. The method is robust and easy to implement. We establish the asymptotic properties of the proposed estimator and contribute to a rigorous theoretical framework for general kernel-weighted sieve M-estimators. Numerical studies corroborate our theoretical results and show that the proposed method performs favorably over existing methods. Applying to a COVID-19 study in Wuhan illustrates the practical utility of our method.

# Main function descriptions

## simAsytransdata
Generate censored data under the transformed hazards model (Box-Cox) with sparse longitudinal covariates for one subject.

Input:

    simAsytransdata
        (
            mu, # the covariate observation process intensity/rate function
            mu_bar, # the largest value of mu
            alpha, # alpha(t): the nonparametric part
            beta, # the regression coefficient vector, the dimension of beta also determines of dimension of covariates
            s, # parameter for Box-Cox transformation 
            cen, # minimum censoring time
            nstep = 20 # the number of steps in the step function Z(t)
        )

Output: Simulated data as a tibble class. 

    X: Censoring time or failure time
    delta: Censoring indicator, 1: not censored
    Covariates: Covariates of dimension d
    obs_times: observation times for covariates
    censoring: censoring time

It is in the [tidy data format](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) that each row represents the observed covariates at one observation time point. The following is an example data generated for one subject:

          X delta covariates[,1]  [,2] obs_times censoring
      <dbl> <lgl>          <dbl> <dbl>     <dbl>     <dbl>
    1 0.235 TRUE          0.0970     1   0.00969         1
    2 0.235 TRUE          0.245      1   0.0611          1
    3 0.235 TRUE          0.245      1   0.0996          1
    4 0.235 TRUE          0.849      1   0.305           1
    5 0.235 TRUE          0.865      1   0.445           1
    6 0.235 TRUE          0.690      1   0.957           1
    7 0.235 TRUE          0.690      1   0.993           1
  


## estproc_ori_d_mul

The implementation of the proposed sieve maximum kernel-weighed log-likelihood estimation (SMKLE) method for censored data with sparse longitudinal covariates. 

Input:

    estproc_ori_d_mul
        (
            data, # Data in the format of data generated from simAsytransdata
            n, # Sample size
            nknots, # Number of knots for B-splines
            norder, # The order of B-splines
            s, # Parameter for Box-Cox transformation 
            h, # Bandwdith
            pl = 0 # Deprecated
        )

Output: A list

    est: Raw estimation results for beta and gamma
    se: Estimated se for $\hat{\beta}$, which is $(\Sigma)^{(-1)}\Omega(\Sigma)^{(-1)}$
    A_est: Raw estimation for $\Sigma$ for the sandwich estimation
    B_est: Raw estimation for $\Omega$ for the sandwich estimation


## estproc_ori_dh
Same to estproc_ori_d_mul but only gives the point estimation. Mainly used for cross-validation function ``hcv``.

## hcv
The wrapper function for the cross-validation method to select bandwidth $h$ for the proposed SMKLE.

Input:

    hcv
        (
            simdata, # Simulated data
            n, # Sample size
            K, # The number of fold for the cross-validation
            s # Parameter for Box-Cox transformation 
        )

Output:
    A data frame consisting of the bandwidth candidates and corresponding CV losses.

## estproc_LVCF_mul
The last value carried forward (LVCF) method for censored data with sparse longitudinal covariates estimated by MLE. 

Input: the same structure as ``estproc_ori_d_mul``. 

Output: the same structure as ``estproc_ori_d_mul``. The results are from LVCF-based MLE. Please see the paper for more details.

## estproc_additive_Cao_mul

The method proposed by Sun, Cao and Chen (2022) for censored data under additive model with sparse longitudinal covariates.

Input: the same structure as ``estproc_ori_d_mul``. 

Output: the same structure as ``estproc_ori_d_mul``.

## estproc_additive_Cao_mul

The method proposed by Cao et al. (2015) for censored data under Cox's model with sparse longitudinal covariates.

Input: the same structure as ``estproc_ori_d_mul``. 

Output: the same structure as ``estproc_ori_d_mul``.

# File descriptions

All the files are standalone and can run without relying on the functions in other files. To achieve this, there are several overlapping functions in the files.

| File names                  | Description                                                   |
|-----------------------------|---------------------------------------------------------------|
| multiple_4_cv_cons          | The proposed SMKLE with bandwidths selected by CV             |
| multiple_4_fixed_cons       | The proposed SMKLE with fixed bandwidths                      |
| multiple_6_LVCF             | LVCF method                                                   |
| multiple_Cao_additive_CV    | Sun, Cao and Chen (2022) with data-driven bandwidth selection |
| multiple_Cao_additive_fixed | Sun, Cao and Chen (2022) with fixed bandwidths                |
| multiple_Cao_Cox_CV         | Cao at el. (2015) with data-driven bandwidth selection        |
| multiple_Cao_Cox_fixed      | Cao at el. (2015) with fixed bandwidths                       |

## References 

* Cao, H., Churpek, M. M., Zeng, D. and Fine, J. P. (2015). Analysis of the proportional hazards model with sparse longitudinal covariates. *Journal of the American Statistical Association* 110 1187–1196.

* Sun, Z., Cao, H. and Chen, L. (2022). *Regression analysis of additive hazards model with sparse longitudinal covariates*. Lifetime Data Analysis 28 263–281.