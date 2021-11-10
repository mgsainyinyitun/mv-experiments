import pandas as pd;
import numpy as np;
from sklearn.cluster import KMeans;
from sklearn.metrics import accuracy_score;


init = "C:\\Users\\Sai Nyi\\Desktop\\pjt\\Experiments\\datasets\\complete-data";

v1 = pd.read_csv(init+"\\Reuters\\01FirstView.csv",header=None);

#v1 = pd.read_csv(init+"\\Caltech\\GISTFeature.csv",header=None);

label = pd.read_csv(init+'\\Reuters\\Label.csv',header=None) # Reuter
#label = pd.read_csv(init+'\\Caltech\\label.csv',header=None);

# NMF
from sklearn.decomposition import NMF;

model = NMF(n_components=10, init='random',max_iter=2000); # 200
W = model.fit_transform(v1);
H = model.components_;

# print('shape of W:',W.shape);
# print('shape of H:',H.shape);

c = len(np.unique(label));
km = KMeans(n_clusters=c); # random center point 
km.fit(v1);

pred = km.labels_+1;
print('accuracy score:', accuracy_score(label, pred));



# Data / xxx
# NMF
# Graph learning 
# Graph fusion
# eigen vector 















