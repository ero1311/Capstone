import numpy as np
import pandas as pd
import pystan as st
import matplotlib.pyplot as plt
import pickle
from hashlib import md5
get_ipython().magic('matplotlib inline')

data=pd.read_csv("btc_usd17-18_gran900.csv")

data=data.head(42*96)
data.to_csv("btc_data.csv")
data_y=np.log(data.high)

mu=np.mean(data_y)
sigma_sq=np.var(data_y)

def cached_model(filename):
    with open(filename, 'rb') as infile:
        code_hash = md5(infile.read()).hexdigest()
    cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = st.StanModel(file=filename)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

def _gem(gma,N):
    prev = 1
    beta_k=np.zeros(N)
    for i in range(N-1):
        beta_k[i] = np.random.beta(1, gma) * prev
        prev -= beta_k[i]
    beta_k[N-1]=prev
    return beta_k

K=3
gamma=2
T=data_y.shape[0]
beta=_gem(gamma,K)

np.random.choice(np.arange(1,K+1),T).shape

stan_data_mappings = {
    'T': T,
    'y': data_y.values.astype(np.floating),
    'K': K,
    'data_mu': mu,
    'data_sigma': sigma_sq,
    'z': np.random.choice(np.arange(1,K+1),T),
    'b': beta,
  }

model=cached_model("hdp_hmm.stan")
fit=model.sampling(data=stan_data_mappings,n_jobs=1)

