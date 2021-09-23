
import numpy as np;
import pandas as pd;

# read data
X = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\papers\\Experiments\\coding\\feature.csv',header=None);
X = X.to_numpy();


from constructW import ConstructW;
from distances import Distances;
from general import General;

gg = General();

dd = Distances();

D = dd.EuDistance(X);



cw = ConstructW();
t = gg.fint_t(X);

G,smpNow,dist,dump,idx,t,W= cw.SimilarityMatrix(X);