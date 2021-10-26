import numpy as np;
from distances import Distances;
from general import General;

class ConstructW():
    
    def __init__(self):
        self.dist = Distances();
        self.gen = General();
        
        
    def __initialize(self,weight_mode):
        if(weight_mode=='binary'):
            self.bBinary = 1;
        else:
            self.bBinary = 0;
            
            
    def construct_complete_graph(self,X,G):
        # L = G.TLG
        Xcomplete = np.dot(G.T,X);
        Xcomplete = np.dot(Xcomplete,G);
        return Xcomplete;
        
    
        
    def SimilarityMatrix(self,X,neighbour_mode='knn',weight_mode='heatkernel',k=5,b_self_connect=0,b_true_knn = 0):
        nSmp = X.shape[0];
        bSpeed = 1;
        maxM = 62500000;
        BlockSize = int(np.floor(maxM/(nSmp*3)));
        
        # Heat Kernel
        t = self.gen.fint_t(X);
        
        
        self.__initialize(weight_mode);
        
        
        if True:
            G = np.zeros((nSmp*(k+1),3));
            dist = [];
            for i in range(1,1+int(np.ceil(nSmp/BlockSize))):
                
                if i == int(np.ceil(nSmp/BlockSize)):
                    smpIdx = np.array([j for j in range((i-1)*BlockSize+1,nSmp+1)]);
                    
                    dist = self.dist.EuDistance(X[smpIdx-1,:],X,0);
                    
                    # get special matrix with NaN other is zeros
                    
                    loc = np.isnan(dist);
                    loc = ~loc;
                    nan_dist = dist.copy();
                    nan_dist[loc] = 0;
                    
                    # end
                    
                    dist = np.nan_to_num(dist,nan=1e100);
                    
                    if bSpeed:
                        nSmpNow = len(smpIdx);
                        dump = np.zeros((nSmpNow,k+1));
                        idx = np.copy(dump);
                        
                        for j in range(1,k+1+1):
                            minimum = np.min(dist,axis=0);
                            dump[:,j-1] = minimum;
                            #idx[:,j-1] = np.in1d(minimum,minimum).nonzero()[0];
                            #idx[:,j-1] = np.where(dist==minimum);
                            
                            idx[:,j-1] = self.__find_row_index_of_min(dist,minimum);
                            
                            dist[dist==minimum] = 1e100;
                    else:
                        # [dump idx ] sort dist
                        # idx = idx[:,1:k+1]
                        # dump = dump(:,1:k+1);
                        pass;
                        
                    if not self.bBinary:
                        #dump = exp(-dump/(2*options.t^2));% ******
                        denominator = 2*np.square(t);
                        dump = np.exp(-dump/denominator);# similarity level 0-1
                 
                    
                    G[(i-1)*BlockSize*(k+1) : nSmp*(k+1), 0] = np.tile(smpIdx-1,k+1);
                    G[(i-1)*BlockSize*(k+1) : nSmp*(k+1), 1] = idx.flatten(order='F');
                
                    if not self.bBinary:
                        G[(i-1)*BlockSize*(k+1) : nSmp*(k+1) , 2] = dump.flatten(order='F'); 
                    else:
                        G[(i-1)*BlockSize*(k+1) : nSmp*(k+1) , 2] = 1; 
                    
                W = self.__construct_W_matrix(G,nSmp);
                
                
                if not b_self_connect:
                    W = W - np.diag(np.diag(W));
                    
                
                # W = self.__perform_base_on_mode(W, X, weight_mode, t);
                
                
                if not b_self_connect:
                    for i in range(nSmp): # 0-49
                        W[i,i] = 0;
                        
                # W = max(W,W');
                # W = self.__get_max_matrix(W, W.T,nSmp);
                W = W+nan_dist;
                W = np.maximum(W,W.T);
                
                        
        return W;
    
    
    def __get_max_matrix(self,X,Y,nSmp):
        W = np.empty((nSmp,nSmp));
        W[:] = 0;
        
        for i in range(nSmp):# 0 to smp-1 (49)
            for j in range(nSmp):
                if(X[i,j] > Y[i,j]):
                    W[i,j] = X[i,j];
                else:
                    W[i,j] = Y[i,j];
        # end for loop
        return W;
                
    
    
    
    
    def __perform_base_on_mode(self,W,X,weight_mode,t):
        if(weight_mode == 'binary'):
            raise('Binary weight can not be used for complete graph!');
        elif(weight_mode == 'heatkernel'):
            W = self.dist.EuDistance(X);
            W = np.exp(-W/(2*np.square(t)));
            
        elif(weight_mode == 'cosine'):
            pass;
        else:
            raise("Invalid Weight Mode");
            
        return W;
        
    
    
    
    
    def __construct_W_matrix(self,G,nSmp):
        # create W full of zeros with size of nSmp x nSmp;
        # for each row
        # assign W to its specific location
        loop = G.shape[0];
        W = np.empty((nSmp,nSmp));
        W[:] = 0;
        
        for i in range(0,loop):
            row = int(G[i,0]);
            col = int(G[i,1]);
            W[row,col] = G[i,2]; 
        return W;
                
        
    
    def __find_row_index_of_min(self,dist,minimum):
        # for each row in dist
        # find index of its minium index
        # assign to an list
        # return
        indices = [];
        nSmp = dist.shape[0];
        for i in range(0,nSmp):
            #index = np.where(dist[i,:]==minimum[i]);
        
            index = dist[:,i].tolist().index(minimum[i]);
            
           
            indices.append(index);
        indices = np.array(indices);
        return indices;
        
        
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    