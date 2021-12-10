# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:05:13 2021

@author: Sai Nyi
"""

import matplotlib.pyplot as plt;

from sklearn.datasets import make_gaussian_quantiles;


X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3);
plt.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k");