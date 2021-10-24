
import pandas as pd;
from randomdeleteview import RandomDeleteViewData;
from Algorithm import Algorithm;
import numpy as np;
rd = RandomDeleteViewData();
al = Algorithm();

v1 = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\01FirstView.csv',header=None);
v2 = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\02SecondView.csv',header=None);
v3 = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\03ThirdView.csv',header=None);
v4 = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\04FourthView.csv',header=None);
v5 = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\05FifthView.csv',header=None);
label = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data\\Reuters\\Label.csv',header=None);

#all_view = [v1.to_numpy(),v2.to_numpy(),v3.to_numpy(),v4.to_numpy(),v5.to_numpy()];

all_view = [v1,v2,v3,v4,v5];

# Randm Remove Data,

Vl = rd.random_remove_data(all_view);


#  {0.0001, 0.001, 0.01, 0.1, 1, 10, 100,1000}
# para1=[.01  1];
# para2=[100,1000,2000];
# para3=[.01];

alpha = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100,1000];
beta = [.01];
gamma = [0.01,1];
lambdas = [100,1000,2000];


#Wx,Hx,Vx,w0 ,Zx= al.implement(Vl, label, alpha[0], beta[0], gamma[0], lambdas[0])
Sx = al.implement(Vl, label, alpha[0], beta[0], gamma[0], lambdas[0])


from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;
km = KMeans(n_clusters=len(np.unique(label)));


km.fit(Sx);
predict = km.labels_;

print(accuracy_score(label,predict))
























