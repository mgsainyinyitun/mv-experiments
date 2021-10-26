# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:25:10 2021

@author: Sai Nyi
"""

import pandas as pd;
import numpy as np;
from Algorithm import Algorithm;
from constructW import ConstructW;
init = "C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data";

v1 = pd.read_csv(init+"\\Caltech\\GISTFeature.csv",header=None);
v2 = pd.read_csv(init+"\\Caltech\\LBPFeature.csv",header=None);
label = pd.read_csv(init+"\\Caltech\\label.csv",header=None);
label = label.to_numpy();
Vl = [v1.to_numpy(),v2.to_numpy()];
cw = ConstructW();

def calculate_laplacian(X):
    degreeM = np.sum(X,axis=1);
    laplacianM = np.diag(degreeM) - X;
    sqrtDegreeM = np.diag(1.0/(degreeM**(0.5)));
    return np.dot(np.dot(sqrtDegreeM,laplacianM),sqrtDegreeM);

ZL = [];
for i in range(len(Vl)):
    Ztemp = cw.SimilarityMatrix(Vl[i]);
    ZL.append(Ztemp.copy());

# averaging
avgG = np.zeros((ZL[0].shape[0]));
for i in range(len(ZL)):
    avgG = avgG + ZL[i];
    avgG = avgG/len(ZL);

L = calculate_laplacian(avgG);
eigVal, eigVec = np.linalg.eig(L);

eigVal = zip(eigVal,range(len(eigVal)));
eigVal = sorted(eigVal,key=lambda eigVal:eigVal[0]);
F = np.vstack([eigVec[:,i] for (v,i) in eigVal[:20]]).T;


from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import normalized_mutual_info_score;
from sklearn.metrics import v_measure_score;

c = len(np.unique(label));
km = KMeans(n_clusters=c);
km.fit(np.real(F));

pred = km.labels_+1;

print('accuracy score:', accuracy_score(label, pred));
















