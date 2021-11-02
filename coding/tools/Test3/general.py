
import numpy as np;
from distances import Distances

class General:
    def __init__(self):
        self.dist = Distances();
        
    
    def incomplete_index_matrix(self,XC,XI):
        """
        input
        XC = Complete X(input)
        XI = Incomplete X (input)
        
        output
        G (Index Matrix);
        G = | 1 , if xc is original instance xi
            | 0, otherwise
            
        """
        # > constuct zeros matrix G of shape nv,n;
        # > loop and update G;
        
        nv = XC.shape[0];
        n = XI.shape[0];
        G = np.zeros((nv,n)); # 45 x 50 => zeros matrix
        
        for i in range(nv):     # 0 to nv-1
            for j in range(n):  # 0 to n-1
                ans = (XC[i]==XI[j]).all();
                if ans:
                    G[i,j] = 1;
            
        return G;
        
        
        
        

    def fint_t(self,X):
        
        # fill NaN to Zero if any
        Xtemp = X.copy();
        Xtemp = np.nan_to_num(Xtemp,nan=0);
        
        D = []
        nSmp = X.shape[0]
        if nSmp > 3000:
            temp = Xtemp[np.random.randint(0,nSmp,(3000)),:]
            D = self.dist.EuDistance(temp)
        else:
            D = self.dist.EuDistance(Xtemp)
        return np.mean(np.mean(D,axis=0))

