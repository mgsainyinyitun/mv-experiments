import numpy as np;

class Distances:
    def EuDistance(self,X,Y=[],bSqrt=1):
        # if Y is empty
    
        aa = 0;
        if len(Y) == 0:
            aa =  np.sum(np.multiply(X,X),axis=1);
            ab =  np.dot(X,X.T); # 3.4 4,3
            D = aa+aa[:,None] - 2*ab;
            D[D<0] = 0;
                
            if bSqrt:
                D = np.sqrt(D);
            return D;
        else:
            aa = np.sum(np.multiply(X,X),axis=1);
            bb = np.sum(np.multiply(Y,Y),axis=1);
            ab = np.dot(X,Y.T);
            #ab = X*(Y.T);
                
            D = aa+bb.T[:,None] - 2*ab;
            D[D<0] = 0;
            if bSqrt:
                D = np.sqrt(D);
            return D;
