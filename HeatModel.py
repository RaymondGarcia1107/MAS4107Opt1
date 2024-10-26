#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:29 2024

@author: ramsonmunoz
"""

from scipy import sparse
import numpy as np

class HeatModel:
    

    def __init__(self,n):
        self.n = n
        
        print()
    
    
    # def __init__(self):
    def create_large_diagonal_matrix(self)->np.ndarray:
        """
        

        Parameters
        ----------
        int : n
            Size of matrix tile.

        Returns
        -------
        n x n Matrix of ones where diagonals have -4.
        
        ex 3 x 3
        
        [[-4 1 1]
         [1 -4 1]
         [1 1 -4]]

        """
        
        return np.ones((self.n,self.n)) - 5*np.eye(self.n)
        
        