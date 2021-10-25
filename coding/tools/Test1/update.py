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
        # for middle loop
        # mid_H=0;
        # for i in range(len(HL)):
        #     if(i == view_no):
        #         continue;
        #     mid_H = mid_H + HL[i]; # (kxn)
        # ##
        # mid_H = alpha*mid_H;
        # print('Shape of midH sum:',mid_H.shape);
        
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
        
    # def update_s(self,w0,lambdas,beta,Z,F): 
    #     # Z = [Z1,Z2,...]; list of all Z in multi-view 
    #     n = Z.shape[0];
        
        
    #     # Pi = np.zeros((n,n));
    #     S = np.zeros((n,n));
        
    #     Zv = self._sum_of_Z(Z,w0); # vo -=> vector (w01,w02,w03,....)
        
    #     for i in range(n):
    #         # Pi = self._solve_p(Pi,F,n,i);
    #         Pi = self._solve_p( F, n, i);
    #         S[:,i] = ( Zv[:,i] - beta*(Pi.T)/4*lambdas ) / 4*np.sum(w0);
        
    #     return S;
            
        
    
    # def update_wo(self,Z,S):
    #     # print(np.linalg.norm(a, 'fro'))
    #     return 1/2*(np.linalg.norm((Z-S),'fro'));
        
    # def solve_f(self,X,c=None,isMax=1,isSym=1):
    #     F,_,_= self.cal_eig.calculate(X,c,isMax,isSym);
    #     return F;
        
        
    # def _solve_p(self,F,n,i):
    #     for j in range(n):
    #         diff = F[i,:] - F[j,:];
    #         Pi[j] = np.square( np.linalg.norm(diff) );
    #     return P;

    # def _sum_of_Z(self,Z,w0):
    #     Zv = np.zeros_like(Z[0]); # n x n 
    #     for i in range(len(Z)):
    #         Zv = Zv + w0[i]*Z[i]; # w0[0] * Z[0] + w0[1] * Z[1] + .... +
    #     return Zv;
            
        
        
        
        
        
        
        
        
        
        
        
        
        
