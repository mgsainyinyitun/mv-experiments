import numpy as np;
from constructW import ConstructW;
# from calculate_eigen import CalculateEigen;
from sklearn.preprocessing import Normalizer;
from update import Update;
import matrix_utils as mu;
class Algorithm():
    
    def __init__(self):
        self.cw = ConstructW();
        #self.cal_eig = CalculateEigen();
        self.transformer = Normalizer();
        self.upd = Update();
        
    def implement(self,VL,label,alpha,beta,gamma,lambdas):
        
        ZL,WL,HL,seed = self.initialize(VL);  
        NVL = self.normilize(VL);
        # c = len(np.unique(label));
        
        LL = []; # for list of lapaclian matrix
        eigValue = [];
        eigVector = [];
        FL = [];
        
        for j in range(len(VL)):
            rel_error = 10;
            normX = mu.norm_fro(NVL[j]);
            while(rel_error > 0.1):
                # print('Size of H',HL[j].shape);
                # print('Size of W',WL[j].shape);
                # print('Size of X',NVL[j].shape);
                # print('Size of Z',ZL[j].shape);
                
                WL[j] = WL[j] * self.upd.update_w(NVL[j], HL[j], WL[j]);
                HL[j] = HL[j] * self.upd.update_h(HL,HL[j], WL[j], NVL[j], ZL[j], j, alpha);
                ZL[j] = ZL[j] * self.upd.update_z(ZL[j], WL[j], gamma);
                
                # ZL[j] =  self.upd.update_zf(WL[j], gamma);
                
                # print('Upper Norm:',mu.norm_fro_err(NVL[j], WL[j], HL[j], normX));
                # print('Norm X :',normX);
                # rel_error = mu.norm_fro_err(A, W, H, norm_A) / norm_A
                rel_error = mu.norm_fro_err(NVL[j], WL[j], HL[j], normX) / normX;
                print("Error is:",rel_error);
                
            # LL.append(self.calculate_laplacian(ZL[j]));
            # value,vector = np.linalg.eig(LL[j]);
            # eigValue.append(value );
            # eigVector.append(vector);
            
            # # x = zip(x, range(len(x)))
            # # x = sorted(x, key=lambda x:x[0])
            # eigValue[j] = zip(eigValue[j],range(len(eigValue[j])));
            # eigValue[j] = sorted(eigValue[j],key=lambda x:x[0]);
            
            # # H = np.vstack([V[:,i] for (v, i) in x[:500]]).T
            # tempF = np.vstack([eigVector[j][:,i] for (v,i) in eigValue[j][:20]]).T;
            # FL.append(tempF.copy());
        
        # averag graph
        avgG = np.zeros((ZL[0].shape[0]));
        for i in range(len(NVL)):
            avgG = avgG + ZL[i];
            avgG = avgG/len(NVL);
        
        L = self.calculate_laplacian(avgG);
        eigVal, eigVec = np.linalg.eig(L);
        
        eigVal = zip(eigVal,range(len(eigVal)));
        eigVal = sorted(eigVal,key=lambda eigVal:eigVal[0]);
        F = np.vstack([eigVec[:,i] for (v,i) in eigVal[:20]]).T;
        
        return WL,HL,ZL,LL,eigValue,eigVector,avgG,F,seed;
                
    def calculate_laplacian(self,X):
        # laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        # return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        degreeM = np.sum(X,axis=1);
        laplacianM = np.diag(degreeM) - X;
        sqrtDegreeM = np.diag(1.0/(degreeM**(0.5)));
        return np.dot(np.dot(sqrtDegreeM,laplacianM),sqrtDegreeM);
        
    
    def calculate_eigen_properties(self,L):
        return np.linalg.eig(L);
        

    
    def initialize(self,V,k=20):
        ZL = [];
        WL = [];
        HL = [];
        np.random.seed(2858947534); # get 0.49
        seed = np.random.get_state()[1][0];
        for i in range(len(V)):
            Ztemp = self.cw.SimilarityMatrix(V[i]);
            Wtemp = np.random.rand(V[i].shape[0],k);
            Htemp = np.random.rand(k,V[i].shape[1]);
            
            ZL.append(Ztemp.copy());
            WL.append(Wtemp.copy());
            HL.append(Htemp.copy());
            
        return ZL,WL,HL,seed;

    
    def normilize(self,V):
        NV = [];
        for i in range(len(V)):
            Vtemp = self.transformer.transform(V[i]);
            NV.append(Vtemp.copy());
        return NV;
  
		















