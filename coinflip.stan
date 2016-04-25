data{
    int N;
    int x[N];
    real prior_width;
}

parameters{
    real<lower=0, upper=1> beta;
}

model{
    beta ~ normal(0.5,prior_width);
    x ~ bernoulli(beta); 
}