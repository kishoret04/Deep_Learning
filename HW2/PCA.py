# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:53:18 2020

@author: Dustin
"""

class PCA:
    """
    
    """
    
    def __init__(self):
        self.__matrix = 'correlation'
        self.__eig_value = None
        self.__eig_vect = None
        self.__ratios = None
        self.__variance = {}
        self.__desire = None
        
    def perform_PCA(self, X):
        
        
        import pandas as pd
        import numpy as np
        
        #Get the type of matrix to complete PCA on
        matrix = ''
        while matrix not in ['correlation', 'covariance']:
            matrix = input('What type of matrix do you want to complete PCA on?\n').lower()
            if matrix == '':
                matrix = self.__matrix
                break
        
        self.__matrix = matrix
        
        #Get the correlation matrix
        if matrix == 'correlation':
            square = X.corr() #Square Matrix by correlation
        elif matrix == 'covariance':
            square = X.cov() #square matrix by covariance
        
        #fill nulls with 0
        nas = square.isnull().sum().sum()
        if nas > 0:
            square = square.fillna(0)
            square.dropna(axis = 0, how = 'all', inplace = True)
            square.dropna(axis = 1, how = 'all', inplace = True)
            print('FYI: there were null values. They were filled with 0. May want to look into this.\n')
        
        #Get eigenvalues and vectors
        eigs = np.linalg.eig(square)
        eig_value = eigs[0]
        eig_vect = eigs[1]
        
        print('Sum of eig vectors squared: {}\n'.format(sum(eig_vect[:,0]**2)))
        print('Sum of eigen values: {}\n'.format(sum(eig_value)))
        
        print('Eigen Values and Vectors saved')
        
        self.__eig_value = eig_value
        self.__eig_vect = eig_vect
        
    def get_ratios(self):
        
        import numpy as np
        import pandas as pd
        
        #Get the variance explained
        v = []
        variance = input('What is the ratio of variance you want explained? (Use Decimal Format)\n')
        while variance != '':
            v += [float(variance)]
            variance = input('What is the next ratio of variance you want explained? (Use Decimal Format)\n')
        
        #Get the variances explained and relative index values
        eig_value = self.__eig_value
        variance = {}

        ratio = np.cumsum(eig_value)/sum(eig_value)            
        for i in v:
            idx = np.where(ratio>i)[0][0] #gets the index
            variance[i] = (ratio[idx], idx)
            print('desired variance was {} but actual was {} with an index of {}\n'.format(i, ratio[idx], idx))
            print('python starts with 0')
        
        self.__ratios = ratio
        self.__variance = variance

    def project(self, train, test, desire):
        
        import numpy as np
        import pandas as pd
        
        
        #Get the index of the ratios
        variance = self.__variance
               
        if desire in variance.keys():
            idx = variance[desire][1]
        else:
            eig_value = self.__eig_value
            if self.__ratios is None:
                ratio = np.cumsum(eig_value)/sum(eig_value)
                self.__ratios = ratio
            else:
                ratio = self.__ratios
            idx = np.where(ratio > desire)[0][0] #gets the index
        
        self.__desire = (desire, idx)
        #project
        eig_vect = self.__eig_vect
        
        split = input('Is the x and y split?\n')
        if 'n' in split:
            print('Sorry you need to have the y values split from the data before projecting')
        else:
            train = pd.DataFrame(np.dot(train,eig_vect))
            test = pd.DataFrame(np.dot(test,eig_vect))
        
        train_proj = train.iloc[:, 0:idx+1]
        test_proj = test.iloc[:, 0:idx+1]
        
        return train_proj, test_proj

    def plot(self):
        
        import matplotlib.pyplot as plt 
        
        ratios = self.__ratios
        eig_value = self.__eig_value
        desire = self.__desire
        
        #plot eig_values
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('Eigenvalues')
        ax.set_ylabel('Eignevalues')
        ax.set_xlabel('Index')
        ax.plot(range(len(eig_value)), eig_value, alpha=0.75)
        ax.scatter(desire[1], eig_value[desire[1]], c='black', alpha=1)
        ax.plot(range(len(eig_value)), [eig_value[desire[1]]]*len(eig_value), 'r--')
        plt.savefig('Eig_Values.png')
        plt.show()
        
        #plot ratios
        fig1, ax1 = plt.subplots(figsize=(12,8))
        ax1.set_title('Ratio of Variance Explained')
        ax1.set_ylabel('Ratio of Variance Explained')
        ax1.set_xlabel('Index')
        ax1.plot(range(len(ratios)), ratios, alpha=0.75)
        ax1.scatter(desire[1], ratios[desire[1]], c='black', alpha=1)
        ax1.plot(range(len(ratios)), [ratios[desire[1]]]*len(ratios), 'r--')
        plt.savefig('Ratio_Values.png')
        plt.show()