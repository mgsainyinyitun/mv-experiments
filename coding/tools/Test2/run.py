# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 15:09:09 2021

@author: Sai Nyi
"""
import pandas as pd;
import numpy as np;
from Algorithm import Algorithm;
init = "C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data";

v1 = pd.read_csv(init+"\\Caltech\\GISTFeature.csv",header=None);
v2 = pd.read_csv(init+"\\Caltech\\LBPFeature.csv",header=None);
label = pd.read_csv(init+"\\Caltech\\label.csv",header=None);
label = label.to_numpy();
Vl = [v1.to_numpy(),v2.to_numpy()];

al = Algorithm();

# 100,1000,2000

alpha = [0.001,0.1,1];
beta = [0.01]
gamma = [0.1]
lambdas = [100,1000,2000];

Wl,Hl,Zl,w0,S,F= al.implement(Vl, label, alpha[0], beta[0], gamma[0], lambdas[0]);

### average eigen

# avg = (Fl[0] + Fl[1]) /2;

from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import normalized_mutual_info_score;
from sklearn.metrics import v_measure_score;

c = len(np.unique(label));
km = KMeans(n_clusters=c); # random center point 
km.fit(F);

pred = km.labels_+1;

print('accuracy score:', accuracy_score(label, pred));
# print('NMI:',normalized_mutual_info_score(label, pred));
# print('v measure score:',v_measure_score(label, pred));
















