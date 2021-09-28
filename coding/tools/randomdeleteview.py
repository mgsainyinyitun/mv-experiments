import pandas as pd
import numpy as np

class RandomDeleteViewData():
    
    
    def remove_nan(self,X):
        # remove all sample with include NaN
        X = X[~np.isnan(X)];
        return X;
        


    def _separate_views(self,X,number_of_complete):
        view_total = len(X)
        X_to_remove = []
        X_not_to_remove = []

        for i in range(number_of_complete):
            X_not_to_remove.append(X[i])
            X.pop(i)
        X_to_remove  = X
        return X_not_to_remove,X_to_remove
    

    def random_remove_data(self,X,percent=10,number_of_complete=0,random_state = None):
        """ 
        X                   =  List data type of all view .e.g X = [View1,View2,View3, ...] , view => (sample x feature)
        percent             =  percentage of data want to randomly remove
        number_of_complete  =  number of view that will not randomly remove , should not greater than number of view
        random_state        =  random state of random number 
        """

        no_of_sample = len(X[0])
        no_of_missing = int(((no_of_sample)/100) * percent)

        X_final  = []

        # if number_of_complete greater than 0 => don't randomly  remove some views

        if number_of_complete>0:
            X_final,X = _separate_views(X,number_of_complete)
        
        if(random_state):
            np.random.seed(random_state)
        
        # for each view, randomly remove items

        for V in X: # X = [V1,V2];
            Vtemp = V.copy(deep=True);
            for i in range(no_of_missing):
                # Generate random number between 0 - sample number
                col_number = len(V.iloc[0])
                to_remove = np.random.randint(0,no_of_sample)
                Vtemp.iloc[to_remove] = [np.NaN for i in range(col_number)]
            X_final.append(Vtemp) # copy of V 
        return X_final




    