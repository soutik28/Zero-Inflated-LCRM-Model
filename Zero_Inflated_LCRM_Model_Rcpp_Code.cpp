// RCPP Code: Zero Inflated LCRM Model

# define ARMA_WARN_LEVEL 0
# include <RcppDist.h>
# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, RcppDist)]]
using namespace Rcpp;
using namespace RcppArmadillo;
// [[Rcpp::plugins(cpp11)]]


// p = number of linear covariates
// q = number of instrumental variables
// n = number of individuals
// m = number of repeated measurements taken on each individual
// N = n*m is the total sample size 


// Function to generate the latent variable r corresponding to the longitudinal circular response theta_{Y}
// [[Rcpp::export]]
arma::vec ryCPP(arma::vec& ry0, arma::mat& wy, arma::mat& X, arma::vec& b1, arma::vec& b2, arma::vec& beta1, arma::vec& beta2){
  // ry0 = N x 1 vector of initial values of the latent variable r corresponding to theta_{Y}
  // wy = N x 2 matrix whose columns consist of (cos(theta_{Y}), sin(theta_{Y}))
  // X = N x (p+3) Stage-I design matrix whose columns consist of (1, linear_covariates, cos(theta_{X}), sin(theta_{X}))
  // b1 = N x 1 vector of random effect corresponding to cos(theta_{Y})
  // b2 = N x 1 vector of random effect corresponding to sin(theta_{Y})
  // beta1 = (p+3) x 1 vector of parameters corresponding to cos(theta_{Y})
  // beta2 = (p+3) x 1 vector of parameters corresponding to sin(theta_{Y})
  
  arma::mat muy = join_horiz(X * beta1 + b1, X * beta2 + b2);  // N x 2 matrix
  arma::vec qy = arma::sum(wy % muy, 1);  // N x 1 vector
  
  arma::vec genunif1 = arma::randu(ry0.n_elem);  // N x 1 vector of random samples generated from Uniform(0,1)
  arma::vec genunif2 = arma::randu(ry0.n_elem);  // N x 1 vector of random samples generated from Uniform(0,1)
  
  arma::vec lns = log(genunif1) - 0.5 * pow(ry0 - qy, 2);
  arma::vec l1 = qy + arma::max(join_horiz(-qy, -sqrt(-2*lns)), 1);
  arma::vec l2 = qy + sqrt(-2*lns);
  arma::vec ry = sqrt((pow(l2, 2) - pow(l1,2)) % genunif2 + pow(l1, 2));  // update the latent variable r corresponding to theta_{Y}
  return(ry);
}


// Function to generate the latent variable r corresponding to the circular covariate theta_{X}
// [[Rcpp::export]]
arma::vec rxCPP(arma::vec& rx0, arma::mat& wx, arma::mat& V, arma::vec& alpha1, arma::vec& alpha2){
  // rx0 = N x 1 vector of initial values of the latent variable r corresponding to theta_{X}
  // wx = N x 2 matrix whose columns consist of (cos(theta_{X}), sin(theta_{X}))
  // V = N x (q+1) Stage-II design matrix whose columns consist of (1, linear_instrumental_variables)
  // alpha1 = (q+1) x 1 vector of parameters corresponding to cos(theta_{X})
  // alpha2 = (q+1) x 1 vector of parameters corresponding to sin(theta_{X})
  
  arma::mat mux = join_horiz(V * alpha1, V * alpha2);  // N x 2 matrix
  arma::vec qx = arma::sum(wx % mux, 1);  // N x 1 vector
  
  arma::vec genunif3 = arma::randu(rx0.n_elem);  // N x 1 vector of random samples generated from Uniform(0,1)
  arma::vec genunif4 = arma::randu(rx0.n_elem);  // N x 1 vector of random samples generated from Uniform(0,1)
  
  arma::vec lnss = log(genunif3) - 0.5 * pow(rx0 - qx, 2);
  arma::vec l11 = qx + arma::max(join_horiz(-qx, -sqrt(-2*lnss)), 1);
  arma::vec l21 = qx + sqrt(-2*lnss);
  arma::vec rx = sqrt((pow(l21, 2) - pow(l11,2)) % genunif4 + pow(l11, 2));  // update the latent variable r corresponding to theta_{X}
  return(rx);
}


// Function to create a matrix from a vector with specific order nrow x ncol
// [[Rcpp::export]]
arma::mat vec2mat(arma::vec x, int nrow, int ncol) {
  arma::mat y(x);
  y.reshape(nrow, ncol);
  return y;
}


// Function to create a matrix of order 2 x 2
// [[Rcpp::export]]
arma::mat createMatrix(double a11, double a12, double a21, double a22) {
  arma::mat Matrix(2, 2);  // Create a 2 x 2 matrix
  
  // fill the matrix with values
  Matrix(0, 0) = a11;
  Matrix(0, 1) = a12;
  Matrix(1, 0) = a21;
  Matrix(1, 1) = a22;
  
  return Matrix;
}


// Function to generate posterior samples of the model parameters of both Stage-I and Stage-II
// [[Rcpp::export]]
arma::mat zilcrmCPP(int iter, int n, int m, arma::vec& rx0, arma::vec& ry0,
                    arma::vec& muA1, arma::vec& muA2, arma::mat& sigA1, arma::mat& sigA2,
                    arma::vec& muB1, arma::vec& muB2, arma::mat& sigB1, arma::mat& sigB2,
                    double shape0, double rate0, double lambda0, double delta, 
                    arma::vec& thetaY, arma::mat& xData, arma::vec& thetaX, arma::mat& vData){
  
  // iter = number of iterations
  // n = number of individuals
  // m = number of repeated measurements taken on each individual
  // rx0 = N x 1 vector of initial values of the latent variable r corresponding to theta_{X}
  // ry0 = N x 1 vector of initial values of the latent variable r corresponding to theta_{Y}
  // muA1, sigA1 = hyperparameters of the prior distribution of alpha_{1} vector
  // muA2, sigA2 = hyperparameters of the prior distribution of alpha_{2} vector
  // muB1, sigB1 = hyperparameters of the prior distribution of beta_{1} vector
  // muB2, sigB2 = hyperparameters of the prior distribution of beta_{2} vector
  // shape0, rate0 = hyperparameters of the prior distribution of s_{2} = (sigma_{2})^2 * (1-rho^2)
  // lambda0 = hyperparameters of the prior distribution of s_{1} = rho * (sigma_{2}/sigma_{1})
  // delta = censoring value
  // thetaY = N x 1 vector of theta_{Y}
  // xData = N x (p+1) matrix whose columns contain (1, linear covariates)
  // thetaX = n x 1 vector of theta_{X}
  // vData = n x (q+1) matrix whose columns contain (1, instrumental variables)
  
  int N = n*m;
  
  // Generate from prior distributions of alpha_{1}, alpha_{2}, beta_{1}, beta_{2}
  arma::vec alp1pr = as<arma::vec>(wrap(rmvnorm(1, muA1, sigA1)));  // prior value of alpha_{1}
  arma::vec alp2pr = as<arma::vec>(wrap(rmvnorm(1, muA2, sigA2)));  // prior value of alpha_{2}
  arma::vec bet1pr = as<arma::vec>(wrap(rmvnorm(1, muB1, sigB1)));  // prior value of beta_{1}
  arma::vec bet2pr = as<arma::vec>(wrap(rmvnorm(1, muB2, sigB2)));  // prior value of beta_{2}
  
  // Generate from prior distributions of s_{1} and s_{2}
  double scale0 = 1/rate0;  // scale hyperparameter of s_{2}
  double s2prSTAR = as<double>(wrap(arma::randg(1, arma::distr_param(shape0, scale0))));  // prior value of tau = 1/s_{2}
  double mn1 = 0;  // mean hyperparameter of s_{1}
  double sd1 = sqrt(1/(s2prSTAR * lambda0));  // standard deviation hyperparameter of s_{1}
  double s1pr = as<double>(wrap(arma::randn(1, arma::distr_param(mn1, sd1)))); // prior value of s_{1}
  double s2pr = 1/s2prSTAR;    // prior value of s_{2}
  double sig1sq = 1/s2pr;   // prior value of s_{1}
  double sig2sq = (pow(s1pr, 2) + pow(s2pr, 2))/s2pr;  // prior value of (sigma_{2})^2
  double rho = s1pr/sqrt(pow(s1pr, 2) + pow(s2pr, 2));  // prior value of rho
  double a12 = rho * sqrt(sig1sq) * sqrt(sig2sq);
  arma::mat SIGbpr = createMatrix(sig1sq, a12, a12, sig2sq);
  arma::mat invSIGbpr = inv_sympd(SIGbpr);
  
  arma::mat invsA1 = inv(sigA1);  // inverse of sigA1
  arma::mat invsA2 = inv(sigA2);  // inverse of sigA2
  arma::mat invsB1 = inv(sigB1);  // inverse of sigB1
  arma::mat invsB2 = inv(sigB2);  // inverse of sigB2
  
  // Store of posterior samples from the full conditional distributions of the parameters
  arma::mat alp1po(iter, alp1pr.n_elem);  // posterior samples of alpha_{1}
  arma::mat alp2po(iter, alp2pr.n_elem);  // posterior samples of alpha_{2}
  arma::mat bet1po(iter, bet1pr.n_elem);  // posterior samples of beta_{1}
  arma::mat bet2po(iter, bet2pr.n_elem);  // posterior samples of beta_{2}
  arma::vec sig2sqpo(iter);  // posterior samples of (sigma_{2})^2
  arma::vec rhopo(iter);  // posterior samples of rho
  
  arma::mat V = vData; // N x (q+1) design matrix of Stage-II
  arma::mat H1 = (trans(V) * V) + invsA1; // (q+1) x (q+1)
  arma::mat H2 = (trans(V) * V) + invsA2; // (q+1) x (q+1)
  arma::mat dispA1 = inv(H1); // (q+1) x (q+1) covariance matrix of the full conditional distribution of alpha_{1}
  arma::mat dispA2 = inv(H2); // (q+1) x (q+1) covariance matrix of the full conditional distribution of alpha_{2}
  
  arma::vec zerothetaX(n);  // latent variable theta_{X}^{*}
  arma::vec zerothetaY(N);  // latent variable theta_{Y}^{*}
  for (int j = 0; j < iter; j++){
    
    // Generation of the latent variable theta_{X}^{*}
    for(int i = 0; i < n; i++){
      if (thetaX(i) == 0) {
        double muthX1 = as<double>(wrap(V.row(i) * alp1pr));
        double muthX2 = as<double>(wrap(V.row(i) * alp2pr));
        double X1 = as<double>(wrap(rtruncnorm(1, muthX1, 1, 0, R_PosInf)));
        double X2 = as<double>(wrap(rtruncnorm(1, muthX2, 1, -X1*tan(delta), X1*tan(delta))));
        zerothetaX(i) = atan2(X2, X1);
      } else
        zerothetaX(i) = thetaX(i);
    }

    arma::mat wthX = join_horiz(cos(zerothetaX), sin(zerothetaX));  // n x 2, column bind of cos(theta_{X}^{*}) and sin(theta_{X}^{*})
    arma::vec rx = rxCPP(rx0, wthX, V, alp1pr, alp2pr);  // n x 1, generation of the latent variable r corresponding to theta_{X}^{*}
    arma::vec x1 = rx % wthX.col(0);  // n x 1, latent variable X_{1}^{*}
    arma::vec x2 = rx % wthX.col(1);  // n x 1, latent variable X_{2}^{*}
    
    arma::vec h1 = (trans(V) * x1) + (invsA1 * muA1); // (q+1) x 1
    arma::vec h2 = (trans(V) * x2) + (invsA2 * muA2); // (q+1) x 1
    arma::vec meanA1 = dispA1 * h1;  // (q+1) x 1 mean vector of the full conditional distribution of alpha_{1}
    arma::vec meanA2 = dispA2 * h2;  // (q+1) x 1 mean vector of the full conditional distribution of alpha_{2}
    
    alp1po.row(j) = as<arma::rowvec>(wrap(rmvnorm(1, meanA1, dispA1)));  // posterior sample of alpha_{1}
    alp2po.row(j) = as<arma::rowvec>(wrap(rmvnorm(1, meanA2, dispA2)));  // posterior sample of alpha_{2}
    
    alp1pr = as<arma::vec>(wrap(alp1po.row(j))); 
    alp2pr = as<arma::vec>(wrap(alp2po.row(j))); 
    rx0 = rx;
    
    // Replicate the elements of cos(theta_{X}^{*}) and sin(theta_{X}^{*})
    NumericVector cosRep = rep_each(as<NumericVector>(wrap(cos(zerothetaX))), m);  // N x 1
    arma::vec costhX = as<arma::vec>(wrap(cosRep));  // N x 1
    NumericVector sinRep = rep_each(as<NumericVector>(wrap(sin(zerothetaX))), m);  // N x 1
    arma::vec sinthX = as<arma::vec>(wrap(sinRep));  // N x 1
    arma::mat wthXRep = join_horiz(costhX, sinthX);  // N x 2
    
    arma::mat X = join_horiz(xData, wthXRep);  // N x (p+3) updated design matrix of Stage-I
    
    // Update the random effect vector b_i = (b1_i, b2_i)
    arma::vec paramb1 = (ry0 % cos(zerothetaY)) - (X * bet1pr);  // N x 1
    arma::vec paramb2 = (ry0 % sin(zerothetaY)) - (X * bet2pr);  // N x 1
    arma::vec nparamb1 = arma::sum(trans(vec2mat(paramb1, m, n)), 1);  // n x 1
    arma::vec nparamb2 = arma::sum(trans(vec2mat(paramb2, m, n)), 1);  // n x 1
    arma::mat idenMat(2 , 2, arma::fill::eye);  // 2 x 2
    arma::mat dispb = inv_sympd((m * idenMat) + invSIGbpr);  // 2 x 2
    arma::mat joinb1b2 = trans(join_horiz(nparamb1, nparamb2));  // 2 x n
    arma::mat meanb(2, n);  // 2 x n
    arma::mat b(n, 2);  // n x 2
    for (int i = 0; i < n; i++){
      meanb.col(i) = dispb * joinb1b2.col(i);  // 2 x 1
      b.row(i) = as<arma::rowvec>(wrap(rmvnorm(1, meanb.col(i), dispb)));  // 1 x 2
    }
    arma::vec b1 = b.col(0);  // n x 1
    arma::vec b2 = b.col(1);  // n x 1
    
    // Replicate the elements of b1_i and b2_i
    NumericVector latb1Rep = rep_each(as<NumericVector>(wrap(b1)), m); // N x 1
    arma::vec lb1 = as<arma::vec>(wrap(latb1Rep)); // N x 1
    NumericVector latb2Rep = rep_each(as<NumericVector>(wrap(b2)), m); // N x 1
    arma::vec lb2 = as<arma::vec>(wrap(latb2Rep)); // N x 1
    
    // Generation of the latent variable theta_{Y}^{*}
    for (int i = 0; i< N; i++){
      if (thetaY(i) == 0){
        double muthY1 = as<double>(wrap(X.row(i) * bet1pr + lb1(i)));
        double muthY2 = as<double>(wrap(X.row(i) * bet2pr + lb2(i)));
        double Y1 = as<double>(wrap(rtruncnorm(1, muthY1, 1, 0, R_PosInf)));
        double Y2 = as<double>(wrap(rtruncnorm(1, muthY2, 1, -Y1*tan(delta), Y1*tan(delta))));
        zerothetaY(i) = atan2(Y2, Y1);
      } else
        zerothetaY(i) = thetaY(i);
    }
    
    arma::mat wthY = join_horiz(cos(zerothetaY), sin(zerothetaY));  // N x 2, column bind of cos(theta_{Y}^{*}) and sin(theta_{Y}^{*})
    arma::vec ry = ryCPP(ry0, wthY, X, lb1, lb2, bet1pr, bet2pr);   // N x 2, generation of the latent variable r corresponding to theta_{Y}^{*}
    arma::vec y1 = ry % wthY.col(0);   // N x 1, latent variable Y_{1}^{*}
    arma::vec y2 = ry % wthY.col(1);   // N x 1, latent variable Y_{2}^{*}
    
    arma::mat G1 = (trans(X) * X) + invsB1;  // (p+3) x (p+3)
    arma::mat G2 = (trans(X) * X) + invsB2;  // (p+3) x (p+3)
    arma::vec g1 = (trans(X) * (y1 - lb1)) + (invsB1 * muB1);  // (p+3) x 1
    arma::vec g2 = (trans(X) * (y2 - lb2)) + (invsB2 * muB2);  // (p+3) x 1
    arma::mat dispB1 = inv(G1);  // (p+3) x (p+3) covariance matrix of the full conditional distribution of beta_{1}
    arma::mat dispB2 = inv(G2);  // (p+3) x (p+3) covariance matrix of the full conditional distribution of beta_{2}
    arma::vec meanB1 = dispB1 * g1;  // (p+3) x 1 mean vector of the full conditional distribution of beta_{1}
    arma::vec meanB2 = dispB2 * g2;  // (p+3) x 1 mean vector of the full conditional distribution of beta_{2}
    
    bet1po.row(j) = as<arma::rowvec>(wrap(rmvnorm(1, meanB1, dispB1)));  // posterior sample of beta_{1}
    bet2po.row(j) = as<arma::rowvec>(wrap(rmvnorm(1, meanB2, dispB2)));  // posterior sample of beta_{2}
    
    bet1pr = as<arma::vec>(wrap(bet1po.row(j))); 
    bet2pr = as<arma::vec>(wrap(bet2po.row(j)));
    ry0 = ry;
    
    // Generation of the posterior sample of s_{1}
    double temps1 = lambda0 + arma::sum(pow(b1, 2));
    double means1 = (arma::sum(b1 % b2))/temps1;   // mean of the full conditional distribution of s_{1}
    double sds1 = sqrt(1/(s2prSTAR * temps1));     // standard deviation of the full conditional distribution of s_{1}
    double s1post = as<double>(wrap(arma::randn(1, arma::distr_param(means1, sds1))));   // posterior sample of s_{1}
    
    // Generation of the posterior sample of s_{2}
    double sh = shape0 + (n+1)/2;   // update the shape parameter
    arma::vec temps2 = b2 - (s1post * b1);
    double ra = rate0 + (0.5 * (arma::sum(pow(temps2, 2)) + (lambda0 * pow(s1post, 2))));  
    double sc = 1/ra;    // update the scale parameter
    double s2postSTAR = as<double>(wrap(arma::randg(1, arma::distr_param(sh, sc))));  // tau
    double s2post = 1/s2postSTAR;    // posterior sample of s_{2}
    
    double sig2sqpost = (pow(s1post, 2) + pow(s2post, 2))/s2post;
    double rhopost = s1post/sqrt(pow(s1post, 2) + pow(s2post, 2));   // posterior sample of rho
    double sig1sqpost = 1/s2post;
    
    // Update the covariance matrix of the random effect b_i
    double a12post = rhopost * sqrt(sig1sqpost) * sqrt(sig2sqpost);
    arma::mat SIGbpost = createMatrix(sig1sqpost, a12post, a12post, sig2sqpost);  // covariance matrix of b_i
    arma::mat invSIGbpost = inv_sympd(SIGbpost);  // inverse of the covariance matrix of b_i
    s2prSTAR = s2postSTAR;
    invSIGbpr = invSIGbpost;
    
    sig2sqpo(j) = as<double>(wrap(sig2sqpost));   // posterior sample of (sigma_{2})^2
    rhopo(j) = as<double>(wrap(rhopost));     // posterior sample of rho
    
  }
  arma::mat Mat1 = join_horiz(bet1po, bet2po);
  arma::mat Mat2 = join_horiz(alp1po, alp2po);
  arma::mat Mat3 = join_horiz(sig2sqpo, rhopo);
  arma::mat M = join_horiz(Mat1, Mat2, Mat3);  // store the posterior samples of the parameters
  
  return M;
}
