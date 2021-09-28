
import numpy as np;
import pandas as pd;

# read data
X = pd.read_csv('C:\\Users\\Sai Nyi\\Desktop\\papers\\Experiments\\coding\\feature.csv',header=None);
X = X.to_numpy();


from constructW import ConstructW;
from distances import Distances;
from general import General;
from randomdeleteview import RandomDeleteViewData;


gg = General();
dd = Distances();
rm = RandomDeleteViewData();


Y = rm.random_remove_data( [pd.DataFrame(X)]);

cw = ConstructW();
Ytemp = Y[0].to_numpy();

W = cw.SimilarityMatrix(Ytemp);