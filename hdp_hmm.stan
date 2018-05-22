data {
    int T;   // # time points (equally spaced)
    real y[T];      // mean corrected return at time t
    int K;  // number of groups
    real data_mu;
    real data_sigma;
    real alpha;
    vector[K] b; // base stickbreaking weights
}

parameters{
    simplex[K] z_ret[T];
    simplex[K] trans[K]; // transition probabilities between states
    vector[K] mu; // emission means
    vector[K] sigma;
    simplex[K] z0;
}
transformed parameters{
    vector[K] init;
    for(k in 1:K){
        init[k]=alpha/K;
    }
}
model{
    int ind=1;
    real max_z=0.0;
    vector[K] mult;
    for (k in 1:K) {
        mu[k] ~ normal(data_mu, 1);
        sigma[k] ~ lognormal(data_sigma,1);
    }
    for(i in 1:K) {
        trans[i] ~ dirichlet(b);
    }
    z0 ~ dirichlet(init);
    for(i in 1:T) {
        if (i == 1) {
            for(j in 1:K){
                mult[j]=0;
                for(k in 1:K){
                    mult[j]+=trans[j,k]*z0[k];
                }
            }
        }
        else{
            for(j in 1:K){
                mult[j]=0;
                for(k in 1:K){
                    mult[j]+=trans[j,k]*z_ret[i-1,k];
                }
            }
        }
        z_ret[i] ~ dirichlet(mult);
        for(k in 1:K){
            if(z_ret[i,k]>max_z){
                max_z=z_ret[i,k];
                ind=k;                
            }
        }
        y[i] ~ normal(mu[ind], sigma[ind]);
    }
}
