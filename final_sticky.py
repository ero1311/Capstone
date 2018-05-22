import numpy as np
import pandas as pd
import pystan as st
import matplotlib.pyplot as plt
import pickle
from hashlib import md5
import seaborn as sns
get_ipython().magic('matplotlib inline')


data=pd.read_csv("btc_data.csv")
data_y = np.array([np.log(data.loc[i].close)-np.log(data.loc[i-1].close) for i in range(1,data.shape[0])])

K = 3
T = data_y.shape[0]
mu = np.mean(data_y)
sigma = np.std(data_y)

plt.scatter(np.arange(T),data_y)
plt.show()


final_sticky_hdp_hmm='''
data {
    int T;   // # time points (equally spaced)
    real y[T];      // mean corrected return at time t
    int K;  // number of groups
    real data_sigma;
}
transformed data{
    matrix[K,K] identity=diag_matrix(rep_vector(1,K));
    vector[K] z0=rep_vector(1.0/K,K);
}

parameters{
    simplex[K] z_ret[T];
    simplex[K] trans[K]; // transition probabilities between states
    vector[K] mu; // emission means
    simplex[K] b;
    real<lower=0.0000000001> sigma[K];
    real<lower=0.0000000001> gama;
    real<lower=0.0000000001> alpha_kappa;
    real<lower=0, upper=1> ro;
}
transformed parameters{
    real kappa=ro*alpha_kappa;
    real alpha=(1-ro)*alpha_kappa;
    row_vector[K] alpha_beta=alpha*to_row_vector(b);
    matrix[K,K] prior_param;
    vector[K] init=rep_vector(gama/K,K);
    for(k in 1:K){
        prior_param[k]=alpha_beta+kappa*identity[k];
    }
    
}
model{
    vector[K] mult;
    vector[K] sums;
    vector[K] lps;
    gama ~ gamma(1,0.01);
    alpha_kappa ~ gamma(2,0.01);
    ro ~ beta(10,1);
    b ~ dirichlet(init);
    mu ~ normal(0, 1);
    sigma ~ lognormal(data_sigma,1);
    for(k in 1:K)
        trans[k] ~ dirichlet(to_vector(prior_param[k]));
    for(i in 1:T) {
        if (i == 1) {
            for(k in 1:K)
                mult[k]=to_row_vector(trans[k])*z0;
        }
        else{
            for(k in 1:K)
                mult[k]=to_row_vector(trans[k])*to_vector(z_ret[i-1]);
        }
        z_ret[i] ~ dirichlet(mult);
        lps = log(z_ret[i]);
        for (k in 1:K) {
          lps[k] = lps[k] + normal_lpdf(y[i] | mu[k], sigma[k]);
        }
        target += log_sum_exp(lps);
    }
}
'''


model_final=st.StanModel(model_code=final_sticky_hdp_hmm)


sticky_hdp_data_mappings= {
    'T': T,
    'y': data_y,
    'K': K,
    'data_sigma': np.var(data_y),
  }


fit_final_chains1_1000=model_final.sampling(data=sticky_hdp_data_mappings,
                     iter=1000,
                     chains=1,
                     verbose=True)
params=fit_final_chains1_1000.extract()


params['trans'].mean(axis=0)


import matplotlib.cm as cm
colors=cm.rainbow(np.linspace(0,1,K))


cluster_prob=np.mean(params['z_ret'],axis=0)
clusters=[]
for cluster in np.argmax(cluster_prob,axis=1):
    clusters.append(colors[cluster])
plt.figure(figsize=(20,10))
for x,y,c in zip(np.arange(T),data_y,clusters):
    plt.scatter(x=x,y=y,color=c,alpha=0.5)


c_1=[]
c_2=[]
c_3=[]
for x,y,c in zip(np.arange(T),data_y,np.argmax(cluster_prob,axis=1)):
    if c==0:
        c_1.append([x,y])
    elif c==1:
        c_2.append([x,y])
    else:
        c_3.append([x,y])
c_1=np.array(c_1)
c_2=np.array(c_2)
c_3=np.array(c_3)

plt.figure(figsize=(20,10))
plt.scatter(c_1[:,0],c_1[:,1],color=colors[0])
plt.show()
plt.figure(figsize=(20,10))
plt.scatter(c_2[:,0],c_2[:,1],color=colors[1])
plt.show()
plt.figure(figsize=(20,10))
plt.scatter(c_3[:,0],c_3[:,1],color=colors[2])
plt.show()



