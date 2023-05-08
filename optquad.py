"""
MATH 197 Z Exercise 1
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 16 April 2023
"""

import numpy as np 
from numpy import *
import math
import sys


def steepest_descent(fun, A, x, b, grad, tol=1e-6, maxit=50000):

    """
	Parameters
	----------
		fun:callable
			objective function
        A:matrix
            input n dimensional matrix
		x: array
			initial points
        b:array
           input n dimensional array
		grad:callable
			gradient of the objective function
		tol:float
			tolerance of the method (default is 1e-10)
		maxit:int
			maximum number of iterationd

	Returns
	-------
		tuple(x,grad_norm,it)
			x:array
				approximate minimizer or last iteration
            it:int
				number of iteration
			grad_norm:float
				norm of the gradient at x
			
	"""

    grad_norm = np.linalg.norm(grad(A,x,b))
    it = 0

    while grad_norm>=tol and it<maxit:
        
        d = b-np.dot(A,x)
        alpha = (np.dot(np.transpose(d),d))/(np.dot(np.transpose(d),np.dot(A,d)))

        x = x + alpha*d
        grad_norm = np.linalg.norm(grad(A,x,b))
        it = it + 1
        

    return x, it, grad_norm

