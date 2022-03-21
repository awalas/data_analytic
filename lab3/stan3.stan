data {
    int<lower=1> N; //Number of observations
    vector[N] y; //Observations
}

parameters {
    real mu;
    real<lower=0> sigma;
}

model {
    //Prior model
    mu ~ normal(0, 1);
    sigma ~ normal(0, 1);

    //Observational model
    y ~ normal(mu, sigma);
}

generated quantities {
   vector[N] y_rep;
   for(i in 1:N) {
       y_rep[i] = normal_rng(mu, sigma);
   }
}