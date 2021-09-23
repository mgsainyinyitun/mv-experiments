
import numpy as np;
from distances import Distances

class General:
    def __init__(self):
        self.dist = Distances()

    def fint_t(self,X):
        D = []
        nSmp = X.shape[0]
        if nSmp > 3000:
            temp = X[np.random.randint(0,nSmp,(3000)),:]
            D = self.dist.EuDistance(temp)
        else:
            D = self.dist.EuDistance(X)
        return np.mean(np.mean(D,axis=0))

