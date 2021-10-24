import numpy as np;
import numpy.matlib;
from constructW import ConstructW;
from calculate_eigen import CalculateEigen;
from sklearn.preprocessing import Normalizer;
from update import Update;
from randomdeleteview import RandomDeleteViewData;
from general import General;
from sklearn.metrics.pairwise import euclidean_distances

# Algorithm
# ---------
class Algorithm():
    
    def __init__(self):
        self.cw = ConstructW();
        self.cal_eig = CalculateEigen();
        self.transformer = Normalizer();
        self.upd = Update();
        self.rm = RandomDeleteViewData();
        self.gen = General();
        
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
        
        VLr = self.remove_to_complete_(VL);
        
        ZL,GL,S,FL,w0 = self.initialize(VL,VLr, label); 
        NV,WL,HL = self.normilize(VLr,k=20);
        
        print('finish Initialize and Normalize');
        #c = len(np.unique(label));
    
        # Input : multiview data with m view , alpha, beta, gamma, lambdas,
        for i in range(10):
            Sold = S;
            # W - (m,k) 
            # H - (k,n) 
            # Z - (n,n)
            DL = [];
            LL = [];
  
            
            for j in range(len(VL)):
                print('Start updating in process;;');
                WL[j] = np.multiply(WL[j],self.upd.update_w(NV[j], HL[j], WL[j],ZL[j]));
                HL[j] = np.multiply(HL[j],self.upd.update_h(HL, HL[j], WL[j],NV[j],j, alpha));
                ZL[j] = np.multiply(ZL[j],self.upd.update_z(WL[j], ZL[j], gamma));
                w0[j] = self.upd.update_wo(FL[j], S);
                    #### for update F #### 
                print('W0 value are',w0);
                
                DL.append(np.diag(sum(ZL[j])))
                LL.append(GL[j].T*( DL[j]-ZL[j])*GL[j]);
                
                #LL.append(DL[j]-ZL[j]);
                #tempF,_,_ = self.cal_eig.calculate(LL[j],6,0);
                #FL.append(tempF);
                    
                    #max_E , avg_E = self.find_error(NV[j],WL[j],HL[j]);
                    #return WL[i],HL[i],NV[i],w0,ZL[i];
                
            # for update F;
            FL = self.upd.solve_f(LL);
    
            ###
            
            S = self.upd.update_s(w0, lambdas, beta, FL );
            
            if self._ismetStopcriterion(Sold,S):
                return S;
        return S;
    
    def remove_to_complete_(self,VL):
        VLr = [];
        for i in range(len(VL)):
            temp = self.rm.remove_nan(VL[i]);
            VLr.append(temp);
        return VLr;
        
    def _ismetStopcriterion(self,Sold,Snew):
        #if ii>5 &( (norm(Z-Zold,'fro')/norm(Zold,'fro') )<1e-3)
        # break
        norm1 = np.linalg.norm((Snew-Sold),'fro');
        norm2 = np.linalg.norm(Sold,'fro');
        ratio = norm1/norm2;
        
        print('Ratio ::',ratio);
        
        if ratio < 1e-3:
            return True;
        return False;
    
    def initialize(self,VL,VLr,label): # V = [V1,V2,V3,....]
        # Initializing Z with k-NN,
        # Initializing , S, F, wv = 1/V.
        
        ZL = [];
        GL = [];    
        n = VL[0].shape[0];
        
        # c = len(np.unique(label));
        for i in range(len(VLr)):
            Ztemp = self.cw.SimilarityMatrix(VLr[i]);
            ZL.append(Ztemp);
            # For index matrix G
            Gtemp = self.gen.incomplete_index_matrix(VLr[i], VL[i]);
            GL.append(Gtemp);
        
        w0 = np.ones(len(VLr))/len(VLr);
        
        S = np.eye(n);
        D = np.diag(sum(S));
        L = D-S;
        FL,_,_ = self.cal_eig.calculate(L,isMax=0)
        
        
        
        
        return ZL,GL,S,FL,w0;

    def find_error(self,X,W,H): # x = (mxn)
        L = W*H; # (mxk) (kxn) => mxn
        error = euclidean_distances(X,L);
        Di = np.diag(error);
        return max(Di),np.average(Di);

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
		















