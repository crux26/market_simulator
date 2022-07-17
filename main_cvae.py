"""
.py version of ./notebooks/CVAE.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import MinMaxScaler

import os
home_path = r'E:\Dropbox\GitHub\market_simulator'
code_path = home_path + r'\notebooks'
os.chdir(code_path)

import base
import cvae
import importlib
importlib.reload(cvae)

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import datetime
from itertools import product
#%%
X, y = make_classification(n_samples=10000,
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
)
rng = np.random.RandomState(2)
X += 0.02 * rng.uniform(size=X.shape)
# X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    # make_moons(n_samples=10000, noise=0.05, random_state=0),
    # make_circles(n_samples=10000, noise=0.2, factor=0.5, random_state=1),
    linearly_separable,
]

for ds in datasets:
    data, conditions = ds
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    # Plot dataset
    plt.scatter(*data.T, c=conditions, s=2)
    plt.title('original data')
    plt.show()

    #%% Train
    '''
    make_cirlces: 1e+4 적절. 너무 많으면 randomness 감소하는 듯. 5e+3은 too random 같음
    '''
    n_epochs = [1e+3, 5e+3] # default: 10000
    n_latents = [2, 8, 14]
    for (n_epoch, n_latent) in product(n_epochs, n_latents):
    # for n_epoch in n_epochs:
        n_epoch = int(n_epoch)
        time_st = datetime.datetime.now()
        generator = cvae.CVAE(n_latent=6, alpha=0.02)
        generator.train(data, data_cond=conditions.reshape(-1, 1),
                        n_epochs=n_epoch)
        time_end = datetime.datetime.now()
        time_took = time_end - time_st
        print(f'took: {time_took}')
        
        #%% gen sample
        '''
        cond: "label"
        '''
        outer_circle_generated = generator.generate(cond=(0,), n_samples=1000)
        inner_circle_generated = generator.generate(cond=(1,), n_samples=1000)
        
        #%%
        plt.scatter(*outer_circle_generated.T, s=2)
        plt.scatter(*inner_circle_generated.T, s=2)
        plt.title(f'n_epoch={n_epoch}, n_latent={n_latent}, took={time_took}')
        plt.show()