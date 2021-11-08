
import pandas as pd;
import numpy as np;
from Algorithm import Algorithm;
from randomdeleteview import RandomDeleteViewData;

init = "C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data";

# Caltech
# v1 = pd.read_csv(init+"\\Caltech\\GISTFeature.csv",header=None);
# v2 = pd.read_csv(init+"\\Caltech\\LBPFeature.csv",header=None);

# Mfeat
v1 = pd.read_csv(init+"\\Mfeat\\mfeat_fou.csv",header=None);
v2 = pd.read_csv(init+"\\Mfeat\\mfeat_pix.csv",header=None);
v1 = v1.drop(columns=76);
v1 = v1.drop(0);
v2 = v2.drop(0);



# label = pd.read_csv(init+"\\Caltech\\label.csv",header=None); # Caltech Label
label = pd.read_csv(init+"\\Mfeat\\label.csv",header=None); # Mfeat Label

label = label.to_numpy();

Vl = [v1.to_numpy().T,v2.to_numpy().T];


al = Algorithm();

# # 100,1000,2000
# k # c # 
# alpha = [0.001,0.1,1];

gamma = [0.00001,]#0.0001]
beta = [0.01]#,0.1]
lambdas = [100]#,1000,2000];

F_list = [];

for i in range(len(gamma)):
    for j in range(len(beta)):
        for k in range(len(lambdas)):
            Wl,Hl,Zl,avgZ,F = al.implement_complete(Vl, label, gamma[i], beta[j], lambdas[k]);
            F_list.append(F);
            
        

# Wl,Hl,Zl,avgZ,F = al.implement_complete(Vl, label, gamma[0], beta[0], lambdas[0]);

from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import normalized_mutual_info_score;
from sklearn.metrics import v_measure_score;

c = len(np.unique(label));

# km = KMeans(n_clusters=c); # random center point 
# km.fit(np.real(Ff));

# pred = km.labels_+1;

# print('accuracy score:', accuracy_score(label, pred));

for m in range(len(F_list)):
    acc = [];
    for i in range(10):
        km = KMeans(n_clusters=c); # random center point 
        km.fit(np.real(F_list[m]));
        pred = km.labels_+1;
        acc.append(accuracy_score(label, pred));
        # print('accuracy score:', accuracy_score(label, pred));
        
    
    print('Average Accuracy:',np.mean(acc));
    print('Standard Deviation:',np.std(acc));









