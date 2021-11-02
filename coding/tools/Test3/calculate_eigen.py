
import numpy as np;

class CalculateEigen():

    def calculate(self,X,c=None, isMax=1 ,isSym = 1):
        # x, y = np.linalg.eigh(a)
        # Return  [ Value , Vector ]
        
        if c==None:
            c = X.shape[0];
        elif c > X.shape[0]:
            c = X.shape[0];
        
        if isSym ==1:
            X = np.maximum(X, X.T);
        
        
        d,v = np.linalg.eigh(X);
        # d -> eigen value
        # v -> eigen vector
        
        if isMax == 0:
            d1 = np.sort(d);        # sort in ascending order
            idx = np.argsort(d);    # get index
        else:
            d1 = np.sort(d)[::-1];      # sort in descending order
            idx = np.argsort(d)[::-1];  # get index
            
            
        # get first 'c' items
        idx1 = idx[:c];
        
        eigval = d[idx1];
        eigvec = v[:,idx1];
        
        eigval_full = d1; 
        
        return eigvec,eigval,eigval_full;
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        