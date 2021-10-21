import numpy as np;
from constructW import ConstructW;
from calculate_eigen import CalculateEigen;
from sklearn.preprocessing import Normalizer;
from update import Update;

# Algorithm
# ---------
class Algorithm():
    
    def __init__(self):
        self.cw = ConstructW();
        self.cal_eig = CalculateEigen();
        self.transformer = Normalizer();
        self.upd = Update();
        
    def implement(self,VL,label,alpha,beta,gamma,lambdas):
        # while not converged do
        # 	for v = 1 to V, do
        # 		update Hv according to (a);
        # 		updata Wv according to (b);
        # 		updata Zv according to (c);
        # 		updata wv according to (6);
        # 	end for
        # 		updata S according to (d);
        # 		updata F by solving (e);
        # Until stop criterion is met.
        
        ZL,S,F,w0 = self.initialize(VL, label);  
        NV,WL,HL = self.normilize(VL,k=20);
        c = len(np.unique(label));
    
        # Input : multiview data with m view , alpha, beta, gamma, lambdas,
        for i in range(200):
            Sold = S;
            # W - (m,k) 
            # H - (k,n) 
            # Z - (n,n)
            for j in range(len(VL)):
                WL[j] = np.multiply(WL[j],self.upd.update_w(NV[j], HL[j], WL[j]));
                HL[j] = np.multiply(HL[j],self.upd.update_h(HL, HL[j], WL[j],NV[j], ZL[j],j, alpha));
                ZL[j] = np.multiply(ZL[j],self.upd.update_z(HL[j], S, ZL[j], w0, lambdas, gamma))
                w0[j] = self.upd.update_wo(ZL[j], S);
                
            S = self.upd.update_s(w0, lambdas, beta, ZL, F);
            D = np.diag(sum(S));
            L = D-S;
            F,_,_ = self.cal_eig.calculate(L,c,0);
            
            if self._ismetStopcriterion(Sold,S):
                return F;
        return F;
                
        
    def _ismetStopcriterion(self,Sold,Snew):
        #if ii>5 &( (norm(Z-Zold,'fro')/norm(Zold,'fro') )<1e-3)
        # break
        norm1 = np.linalg.norm((Snew-Sold),'fro');
        norm2 = np.linalg.norm(Sold,'fro');
        ratio = norm1/norm2;
        if ratio < 1e-3:
            return True;
        return False;
    
    def initialize(self,V,label): # V = [V1,V2,V3,....]
        # Initializing Z with k-NN,
        # Initializing , S, F, wv = 1/V.
        
        ZL = [];
        n = V[0].shape[0];
        c = len(np.unique(label));
        for i in range(len(V)):
            Ztemp = self.cw.SimilarityMatrix(V[i]);
            ZL.append(Ztemp);
        
        w0 = np.ones(len(V))/len(V);
        
        S = np.eye(n);
        D = np.diag(sum(S));
        L = D-S;
        F,_,_ = self.cal_eig.calculate(L,c,0);
        
        return ZL,S,F,w0;

    
    def normilize(self,V,k=10):
        # V1 = (m x n);
        # for v = 1 to V, do
        # 	Normalize XV
        # 	Initialize HV,WV,
        # end for
        
        NV = [];
        WL = [];
        HL = [];
      
        for i in range(len(V)):
            # Normalize
            sh = V[i].shape; # (m , n)
            Vtemp = self.transformer.transform(V[i]);
            Wtemp = np.matlib.rand(sh[0],k); # (m x k)
            Htemp = np.matlib.rand(sh[1],k).T; # (k x n) 
            NV.append(Vtemp);
            WL.append(Wtemp);
            HL.append(Htemp);

        return NV,WL,HL;
  
# output : Z, S, F.
		















