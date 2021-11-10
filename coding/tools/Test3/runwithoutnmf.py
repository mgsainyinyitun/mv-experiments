# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:35:07 2021

@author: Sai Nyi
"""

import pandas as pd;
import numpy as np;
from Algorithm import Algorithm;
from randomdeleteview import RandomDeleteViewData;
init = "C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data";

# Reuter
v1 = pd.read_csv(init+"\\Reuters\\01FirstView.csv",header=None);
v2 = pd.read_csv(init+'\\Reuters\\02SecondView.csv',header=None);
v3 = pd.read_csv(init+"\\Reuters\\03ThirdView.csv",header=None);
v4 = pd.read_csv(init+"\\Reuters\\05FifthView.csv",header=None);
v5 = pd.read_csv(init+"\\Reuters\\05FifthView.csv",header=None);



label = pd.read_csv(init+'\\Reuters\\Label.csv',header=None) # Reuter
label = label.to_numpy();

Vl = [  v1.to_numpy(), v2.to_numpy(), v3.to_numpy(),v4.to_numpy(),v5.to_numpy() ];

# graph
from constructW import ConstructW;
cw = ConstructW();

gl = [];

for i in range(len(Vl)):
    gl.append(cw.SimilarityMatrix(Vl[i]));

# average graph
avgG = 0;
for i in range(len(gl)):
    avgG = avgG+gl[i];

avgG = avgG/len(avgG);


# eigen 

from update import Update;
upd = Update();
F = upd.solve_f(avgG, 10);



# k-mean

from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;

c = len(np.unique(label));

km = KMeans(n_clusters=c); # random center point 
km.fit(F);
pred = km.labels_+1;
print('accuracy score:', accuracy_score(label, pred));





# spectral cluster
from sklearn.cluster import SpectralClustering

# average all view
avgV=0;
for i in range(len(Vl)):
    avgV = avgV + Vl[i];
avgV=avgV/len(Vl);



clustering = SpectralClustering(n_clusters=c).fit(avgV);
pred = clustering.labels_+1;

print('spectral cluster accuracy score:', accuracy_score(label, pred));













