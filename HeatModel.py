#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:29 2024

@author: ramsonmunoz
"""

from scipy import sparse
import numpy as np
import sys

class HeatModel:
    

    def __init__(self,n):
        self.n = n
    
    
    def create_large_diagonal_matrix(self):
        """
        This function generates B from the assignment document

        Parameters
        ----------
        int : n
            Size of matrix tile.

        Returns
        -------
        n x n Sparse Matrix of ones where diagonals have -4 on main positive diagonal and 1 on offset diagonals.
        
        ex 3 x 3
        
        [[-4 1 0]
         [1 -4 1]
         [0 1 -4]]
        
        ex 4 x 4
        
        [[-4.  1.  0.  0.]
         [ 1. -4.  1.  0.]
         [ 0.  1. -4.  1.]
         [ 0.  0.  1. -4.]]
        
        
        ex 10 x 10
        
        [[-4.  1.  0.  0.  0.  0.  0.  0.  0.]
         [ 1. -4.  1.  0.  0.  0.  0.  0.  0.]
         [ 0.  1. -4.  1.  0.  0.  0.  0.  0.]
         [ 0.  0.  1. -4.  1.  0.  0.  0.  0.]
         [ 0.  0.  0.  1. -4.  1.  0.  0.  0.]
         [ 0.  0.  0.  0.  1. -4.  1.  0.  0.]
         [ 0.  0.  0.  0.  0.  1. -4.  1.  0.]
         [ 0.  0.  0.  0.  0.  0.  1. -4.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  1. -4.]]

        """
        try:
            
            if(self.n < 2):
                raise ValueError("Ensure model size is at least 2")
        
            data =[np.ones(self.n - 1),-4*np.ones(self.n),np.ones(self.n - 1)]
            offsets = [-1,0,1]
        
            return sparse.diags(data,offsets,shape=(self.n,self.n))
        
        except ValueError as e:
            print(e)
            sys.exit()
        
        