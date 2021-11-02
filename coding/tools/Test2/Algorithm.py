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
        
    def implement(self,VicL,VcL,label,alpha,beta,gamma,lambdas):
        
        ZL,WL,HL,GL,w0,F,S = self.initialize(VicL,VcL,k=20);  
        NVL = self.normilize(VcL);
        c = len(np.unique(label));
        
        # for list of lapaclian matrix
        # eigValue = [];
        # eigVector = [];
        # FL = [];
        
        for j in range(len(VcL)):
            rel_error = 10;
            normX = mu.norm_fro(NVL[j]);
            while(rel_error > 0.48):
                
                WL[j] = WL[j] * self.upd.update_w(NVL[j], HL[j], WL[j],ZL[j]);
                HL[j] = HL[j] * self.upd.update_h(HL,HL[j], WL[j], NVL[j], ZL[j], j, alpha);
                # ZL[j] = ZL[j] * self.upd.update_z(ZL[j], WL[j], gamma);            
                # ZL[j] =  self.upd.update_zf(WL[j], gamma);
                
                rel_error = mu.norm_fro_err(NVL[j], WL[j], HL[j], normX) / normX;
                print("Error is:",rel_error);
        
        

        for j in range(len(ZL)):
            ratio = 1e3;
            while ratio > 45.5e-3:
                ZL[j] = (ZL[j] + ZL[j].T)/2;
                ZLold = ZL[j];
                # ZL[j] = ZL[j] * self.upd.update_z(ZL[j], WL[j], gamma); 
                ZL[j] = ZL[j] * self.upd.update_zf(WL[j], gamma);
                ratio = self.calculate_ratio_of_s(ZLold, ZL[j]);
                print('Ratio for ZL of: ',j,' is :',ratio);
            
        # G.T * Z * G;
        
        for j in range(len(ZL)):
            ZL[j] =   np.dot( np.dot(GL[j].T,ZL[j])  , GL[j] ) ; 
        
        # # End for all View update for H,W,Z;
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

        
        return WL,HL,ZL,GL,w0,S,F;
    
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
        
    
    def initialize(self,VicL,VcL,k=20):
        ZL = [];
        WL = [];
        HL = [];
        GL = []; 
        # np.random.seed(2858947534); # get 0.49
        # seed = np.random.get_state()[1][0];
        for i in range(len(VcL)):
            Ztemp = self.cw.SimilarityMatrix(VcL[i],weight_mode='binary');
            
            Wtemp = np.random.rand(VcL[i].shape[0],k);
            Htemp = np.random.rand(k,VcL[i].shape[1]);
            Gtemp = self.gen.incomplete_index_matrix(VcL[i], VicL[i])
            
            ZL.append(Ztemp.copy());
            WL.append(Wtemp.copy());
            HL.append(Htemp.copy());
            GL.append(Gtemp.copy());
        
        w0 = np.ones(len(VcL))/len(VcL);
        
        S = np.eye(VicL[0].shape[0]);
        F = self.upd.solve_f(S, 7);
        
        return ZL,WL,HL,GL,w0,F,S;

    
    def normilize(self,V):
        NV = [];
        for i in range(len(V)):
            Vtemp = self.transformer.transform(V[i]);
            NV.append(Vtemp.copy());
        return NV;
  
		















