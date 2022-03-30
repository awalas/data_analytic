data {
    int<lower=1> N; // Number of data
    int<lower=1> M; // Number of covariates
    matrix[M,N] X;
    array[N] real y;
}

parameters {
   vector[M] beta;
   real alpha;
   real<lower=0> sigma;
}

model {
    // No priors on the slopes
    alpha ~ normal(0,2);
    sigma ~ normal(0,2);

    y ~ normal(X' * beta + alpha, sigma);
}