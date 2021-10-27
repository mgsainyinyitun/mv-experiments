import numpy as np;

# X => (m x n )
# W => (m x k)
# H => (k x n)
# Z => (n x n)

# from calculate_eigen import CalculateEigen;

class Update():
    
    # def __init__(self):
    #     self.cal_eig = CalculateEigen();
        
    
    def update_h(self,HL,H,W,X,Z,view_no,alpha):
        upper = 2*(np.dot(W.T,X));   #   2*(W.T)(X) + 4*HF*Z # (kxm)(mxn) + (kxn)(nxn) => (kxn) 
        lower = 2*( np.dot( np.dot(W.T,W) ,H) );
        return upper/lower; # (k,n) matrix
        
    
    def update_w(self,X,H,W):
        # 2 *X * (H.T);         # (m,n)(n,k)
        up = 2*(np.dot(X,H.T));
        # W * H * (H.T)    # (m,k)(k,n)(n,k)
        down = np.dot(np.dot(W,H),H.T);
        return up/down; # (m,k) matrix
    
    def update_z(self,Z,W,gamma):
        upper = np.dot(W,W.T);
        lower = np.dot(np.dot(Z,W),W.T) + gamma/2;
        return upper/lower; # (n , n) matrix
    
    def update_zf(self,W,gamma):
        I = np.eye(W.shape[0]);
        part = np.linalg.inv(np.dot(W,W.T));
        return I - (gamma/2)*part;
        
    def update_s(self,w0,lambdas,beta,ZL,F,S): 
        # Z = [Z1,Z2,...]; list of all Z in multi-view 
        n = ZL[0].shape[0];
        Zv = self._sum_of_Z(ZL,w0); # vo -=> vector (w01,w02,w03,....)
        
        for i in range(n):
            # Pi = self._solve_p(Pi,F,n,i);
            Pi = self._solve_pi( F, n, i);
            S[:,i] = ( Zv[i,:] - (beta*(Pi.T))/(4*lambdas) ) / (4*np.sum(w0));
        
        return S;
            
        
    def update_w0(self,Z,S):
        fob = np.linalg.norm((S-Z),'fro');
        fob = fob**2;
        return 0.5*np.sqrt(fob);
        
    # def update_wo(self,Z,S):
    #     # print(np.linalg.norm(a, 'fro'))
    #     return 1/2*(np.linalg.norm((Z-S),'fro'));
        
    
    # first version
    def solve_f(self,X,c):
        L = self.calculate_normalized_laplacian(X);
        # D = np.sum(X,axis=0);
        # D = np.diag(D);
        # L = D-X;
        
        eigVal, eigVec = np.linalg.eig(L);
        eigVal = zip(eigVal,range(len(eigVal)));
        eigVal = sorted(eigVal,key=lambda eigVal:eigVal[0]);
        F = np.vstack([eigVec[:,i] for (v,i) in eigVal[:c]]).T;
        print('Size of F:',F.shape);
        return F;

    # def solve_f(self,X,c):
    #     D = np.sum(X,axis=0);
    #     D = np.diag(D);
        
        
    def _solve_pi(self,F,n,i):
        Pi = np.zeros(F.shape[0]);
        for j in range(n):
            diff = F[i,:] - F[j,:];
            Pi[j] = np.square( np.linalg.norm(diff) );
        return Pi;

    def _sum_of_Z(self,Z,w0):
        Zv = np.zeros_like(Z[0]); # n x n 
        for i in range(len(Z)):
            Zv = Zv + w0[i]*Z[i]; # w0[0] * Z[0] + w0[1] * Z[1] + .... 
        return Zv;
    
    
    def calculate_normalized_laplacian(self,X):
        # laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix
        # sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
        # return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)
        degreeM = np.sum(X,axis=1);
        laplacianM = np.diag(degreeM) - X;
        sqrtDegreeM = np.diag(1.0/(degreeM**(0.5)));
        return np.dot(np.dot(sqrtDegreeM,laplacianM),sqrtDegreeM);
            
        
        
# Note
# ---------------
# S = np.eye(n); 
# S = (S+S.T)/2;
# D = np.diag(S);
# L = D - Z;
# 
        
        
        
        
        
        
        
        
        
        
