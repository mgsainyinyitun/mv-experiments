import numpy as np;

# X => (m x n )
# W => (m x k)
# H => (k x n)
# Z => (n x n)

from calculate_eigen import CalculateEigen;

class Update():
    
    def __init__(self):
        self.cal_eig = CalculateEigen();
        
    
    def update_h(self,HL,HF,W,X,Z,view_no,alpha):
        # for middle loop
        mid_H=0;
        for i in range(len(HL)):
            if(i == view_no):
                continue;
            mid_H = mid_H + H[i]; # (kxn)
        ##
        mid_H = alpha*mid_H;
        
        upper = 2*(W.T)(X) + 4*HF*Z # (kxm)(mxn) + (kxn)(nxn) => (kxn)
        lower = (2*W.T*(W)*HF) + mid_H + 2*HF + 2*HF*(Z.T)*Z # (kxm)(mxk)(kxn) + (kxn) + (kxn) (nxn)(nxn)
        return upper/lower; # (k,n) matrix
        
    
    def update_w(self,X,H,W):
        up = X * (H.T);         # (m,n)(n,k)
        down = W * H * (H.T)    # (m,k)(k,n)(n,k)
        final = up/down
        return final; # (m,k) matrix
    
    def update_z(self,H,S,Z,wo,lambdas,gamma):
        upper = (H.T)*H + lambdas * wo * S; # (nxk)(kxn) + (nxn) => (nxn)
        lower = (Z)*(H.T)*(H) + lambdas*wo*Z + gamma/2; # (nxn)(nxk)(kxn) + (nxn) => (nxn);
        final = upper / lower;
        return final; # (n , n) matrix
        
    def update_s(self,w0,lambdas,beta,Z,F): 
        # Z = [Z1,Z2,...]; list of all Z in multi-view 
        n = Z.shape[0];
        
        
        # Pi = np.zeros((n,n));
        S = np.zeros((n,n));
        
        Zv = self._sum_of_Z(Z,w0); # vo -=> vector (w01,w02,w03,....)
        
        for i in range(n):
            # Pi = self._solve_p(Pi,F,n,i);
            Pi = self._solve_p( F, n, i);
            S[:,i] = ( Zv[:,i] - beta*(Pi.T)/4*lambdas ) / 4*np.sum(w0);
        
        return S;
            
        
    
    def update_wo(self,Z,S):
        # print(np.linalg.norm(a, 'fro'))
        return 1/2*(np.linalg.norm((Z-S),'fro'));
        
    def solve_f(self,X,c=None,isMax=1,isSym=1):
        F,_,_= self.cal_eig.calculate(X,c,isMax,isSym);
        return F;
        
        
    def _solve_p(self,F,n,i):
        for j in range(n):
            diff = F[i,:] - F[j,:];
            Pi[j] = np.square( np.linalg.norm(diff) );
        return P;

    def _sum_of_Z(self,Z,w0):
        Zv = np.zeros_like(Z[0]); # n x n 
        for i in range(len(Z)):
            Zv = Zv + w0[i]*Z[i]; # w0[0] * Z[0] + w0[1] * Z[1] + .... +
        return Zv;
            
        
        
        
        
        
        
        
        
        
        
        
        
        
