################# R Code: Zero Inflated LCRM Model #################

## Required packages
library(fBasics)
library(mvtnorm)
library(truncnorm)
library(circular)
library(HDInterval)

#### Data generation ####
## Parameters
delta = 0.035  # delta
beta1 = c(8.3, 4.6, 6.5, -5.1)  # beta_{1} vector
beta2 = c(-1.3, 0.8, 2.1, 2.4)  # beta_{2} vector
alpha1 = c(-6.4, 4.5)  # alpha_{1} vector
alpha2 = c(1.8, -0.8)  # alpha_{2} vector
rho = 0.5  # rho
varb2 = 2  # sigma_{2}^2
varb1 = 1/(varb2 * (1 - rho^2))  # sigma_{1}^2
Sigb = matrix(c(varb1, rho * sqrt(varb1) * sqrt(varb2), rho * sqrt(varb1) * sqrt(varb2), varb2), 2, 2)  # Sigma_{b} matrix

## Sample size
n = 100  # number of individuals
m = 3  # number of repeated measurements taken on each individual
N = n*m  # total sample size

## Store the data
v = rep(0, n)  # linear instrumental variable
x = matrix(0, n, m)  # linear covariate
b1 = rep(0, n); b2 = rep(0, n)  # random effects
thetaX = rep(0, n)  # circular covariate
thetaY = matrix(0, n, m)  # circular response
thXzero = rep(0, n)  # latent variable theta_{X}^{*}
thYzero = matrix(0, n, m)  # latent variable theta_{Y}^{*}

## Generating zero inflated circular covariate as well as longitudinal circular response
set.seed(100)  # setting the seed
for(i in 1:n){
  v[i] = rnorm(1, 2, 1)
  V = cbind(1, v[i])  # design matrix of Stage-II
  x1 = rnorm(1, V %*% vec(alpha1), 1)
  x2 = rnorm(1, V %*% vec(alpha2), 1)
  thetaX[i] = circular(atan2(x2, x1))
  
  x[i,] = rmvnorm(1, rep(0,m), diag(m))
  X = cbind(1, x[i,], cos(thetaX[i]), sin(thetaX[i]))  # design matrix of Stage-I
  b = rmvnorm(1, c(0,0), Sigb)  # vector of random effects
  b1[i] = b[1,1]; b2[i] = b[1,2];
  y1 = rmvnorm(1, X %*% vec(beta1) + b1[i], diag(m))
  y2 = rmvnorm(1, X %*% vec(beta2) + b2[i], diag(m))
  thetaY[i,] = circular(atan2(y2, y1))
  
  thXzero[i] = ifelse(thetaX[i] < delta & thetaX[i] > -delta, 0, thetaX[i])
  for(j in 1:m){
    thYzero[i,j] = ifelse(thetaY[i,j] < delta & thetaY[i,j] > -delta, 0, thetaY[i,j])
  }
}
vData = cbind(1, c(v))  # design matrix of Stage-II
xData = cbind(1, c(t(x)))  # design matrix of Stage-I excluding the circular covariate
thXzero = circular(thXzero)  # vector of latent variable theta_{X}^*
thYzero = circular(c(t(thYzero)))  # vector of latent variable theta_{Y}^*

mean(ifelse(thXzero == 0, 1, 0))  # proportion of zeros in the circular covariate
mean(ifelse(thYzero == 0, 1, 0))  # proportion of zeros in the longitudinal circular response


## First run the Rcpp code with file name "Zero_Inflated_LCRM_Model_Rcpp_Code.cpp"

#### Generating posterior samples from the full conditional distributions #####
iter = 100000  # total number of iterations
burn = 40000  # burn-in samples
lw = 40010  # index of the lower sample point
up = 100000  # index of the upper sample point
ev = 10  # thinning

## Specification of the latent variables and the hyperparameters
rx0 = rep(1, n)
ry0 = rep(1, N)
muA1 = vec(rep(0, length(alpha1)))
sigA1 = diag(100, length(alpha1))
muA2 = vec(rep(0, length(alpha2)))
sigA2 = diag(100, length(alpha2))
muB1 = vec(rep(0, length(beta1)))
sigB1 = diag(100, length(beta1))
muB2 = vec(rep(0, length(beta2)))
sigB2 = diag(100, length(beta2))
shape0 = 1
rate0 = 0.01
lambda0 = 1
  
## Generation of posterior samples of the model parameters
postSamp = zilcrmCPP(iter, n, m, rx0, ry0, muA1, muA2, sigA1, sigA2, muB1, muB2, sigB1, sigB2,
                     shape0, rate0, lambda0, delta = delta, thYzero, xData, thXzero, vData)  # Rcpp code

true = c(beta1, beta2, alpha1, alpha2, varb2, rho)  # vector of true parameter values
s = seq(lw, up, ev)  # sequence of indices of the posterior samples after burn-in and thinning

## Estimates
MEAN = apply(postSamp[s,], 2, "mean")  # vector of posterior mean
BIAS = MEAN - true  # vector of posterior bias
RB = BIAS/true  # # vector of posterior mean relative bias
VAR = apply(postSamp[s,], 2, "var")  # # vector of posterior variance
SE = sqrt(VAR + BIAS^2)  # # vector of posterior standard error
CRED = matrix(apply(postSamp[s,], 2, "hdi"), ncol = 2, byrow = T) # vector of highest posterior density credible interval (HPDCI)

ESTIM = cbind(round(cbind(MEAN, RB, SE, Lower_HPDCI = CRED[,1], Upper_HPDCI = CRED[,2]), 3)) # store the estimates from a single dataset
rownames(ESTIM) = c("beta_{10}", "beta_{11}", "beta_{1C}", "beta_{1S}", "beta_{20}", "beta_{21}", "beta_{2C}", "beta_{2S}", 
                    "alpha_{10}", "alpha_{11}", "alpha_{20}", "alpha_{21}", "sigma_{2}^2", "rho")
cbind(Parameter = true, ESTIM)  # print the summary table

