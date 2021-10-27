import numpy as np;
from constructW import ConstructW;
from sklearn.preprocessing import Normalizer;
from update import Update;
import matrix_utils as mu;
from general import General;

class Algorithm():
    
    def __init__(self):
        self.cw = ConstructW();
        self.transformer = Normalizer();
        self.upd = Update();
        self.gen = General();
        
    def implement(self,VL,label,alpha,beta,gamma,lambdas):
        
        ZL,WL,HL,w0,F,S = self.initialize(VL,k=20);  
        NVL = self.normilize(VL);
        # c = len(np.unique(label));
        
        # for list of lapaclian matrix
        # eigValue = [];
        # eigVector = [];
        # FL = [];
        
        for j in range(len(VL)):
            rel_error = 10;
            normX = mu.norm_fro(NVL[j]);
            while(rel_error > 0.1):
                
                WL[j] = WL[j] * self.upd.update_w(NVL[j], HL[j], WL[j]);
                HL[j] = HL[j] * self.upd.update_h(HL,HL[j], WL[j], NVL[j], ZL[j], j, alpha);
                # ZL[j] = ZL[j] * self.upd.update_z(ZL[j], WL[j], gamma);            
                # ZL[j] =  self.upd.update_zf(WL[j], gamma);
                
                rel_error = mu.norm_fro_err(NVL[j], WL[j], HL[j], normX) / normX;
                print("Error is:",rel_error);
        
        # for j in range(len(VL)):
        #     rel_error = 10;
        #     normX = mu.norm_fro(NVL[j]);
        #     while(rel_error > 0.1):
        #         HL[j] = HL[j] * self.upd.update_h(HL,HL[j], WL[j], NVL[j], ZL[j], j, alpha);
        #         #WL[j] = WL[j] * self.upd.update_w(NVL[j], HL[j], WL[j]);
        #         rel_error = mu.norm_fro_err(NVL[j], WL[j], HL[j], normX) / normX;
        #         print("Error is:",rel_error);
        
        

        for j in range(len(ZL)):
            ratio = 1e3;
            while ratio > 45e-3:
                ZL[j] = (ZL[j] + ZL[j].T)/2;
                ZLold = ZL[j];
                ZL[j] = ZL[j] * self.upd.update_z(ZL[j], WL[j], gamma); 
                ratio = self.calculate_ratio_of_s(ZLold, ZL[j]);
                print('Ratio for ZL of: ',j,' is :',ratio);
            
                
        # End for all View update for H,W,Z;
        count = 0;
        ratio = 1e3;
        while ratio > 1e-3 and count<5:
            #   Z(find(Z<0))=0; % Z less than 0
            print('Value of count',count);
            count = count +1;
            S[S<0] = 0;
            S = (S+S.T)/2;
            S_old = S.copy();
            
            for i in range(len(ZL)):
                w0[i] = self.upd.update_w0(ZL[i], S);
                
            print(w0);

            F = self.upd.solve_f(S, 7);
            # update S;
            S = self.upd.update_s(w0, lambdas, beta, ZL, F, S);
            ratio = self.calculate_ratio_of_s(S_old, S);
            print('Ratio:',ratio);
            
        # averag graph
        # avgG = np.zeros((ZL[0].shape[0]));
        # for i in range(len(NVL)):
        #     avgG = avgG + ZL[i];
        #     avgG = avgG/len(NVL);
        
        # # ....................................................
        
        # L = self.calculate_laplacian(avgG);
        # eigVal, eigVec = np.linalg.eig(L);
        # eigVal = zip(eigVal,range(len(eigVal)));
        # eigVal = sorted(eigVal,key=lambda eigVal:eigVal[0]);
        # F = np.vstack([eigVec[:,i] for (v,i) in eigVal[:20]]).T;
        
        return WL,HL,ZL,w0,S,F;
    
    def calculate_ratio_of_s(self,S_old,S_new):
        diff = (S_new-S_old);
        ratio = np.linalg.norm(diff,'fro') / np.linalg.norm(S_old,'fro');
        return ratio;
                
    def calculate_laplacian(self,X):
        # laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        # return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        degreeM = np.sum(X,axis=1);
        laplacianM = np.diag(degreeM) - X;
        sqrtDegreeM = np.diag(1.0/(degreeM**(0.5)));
        return np.dot(np.dot(sqrtDegreeM,laplacianM),sqrtDegreeM);
        
    
    def initialize(self,V,k=20):
        ZL = [];
        WL = [];
        HL = [];
        # GL = []; 
        # np.random.seed(2858947534); # get 0.49
        # seed = np.random.get_state()[1][0];
        for i in range(len(V)):
            Ztemp = self.cw.SimilarityMatrix(V[i],weight_mode='binary');
            
            Wtemp = np.random.rand(V[i].shape[0],k);
            Htemp = np.random.rand(k,V[i].shape[1]);
            
            ZL.append(Ztemp.copy());
            WL.append(Wtemp.copy());
            HL.append(Htemp.copy());
        
        w0 = np.ones(len(V))/len(V);
        
        S = np.eye(V[0].shape[0]);
        F = self.upd.solve_f(S, 7);
        
        return ZL,WL,HL,w0,F,S;

    
    def normilize(self,V):
        NV = [];
        for i in range(len(V)):
            Vtemp = self.transformer.transform(V[i]);
            NV.append(Vtemp.copy());
        return NV;
  
		















