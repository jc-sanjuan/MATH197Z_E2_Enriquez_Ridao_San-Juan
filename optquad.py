"""
MATH 197 Z Exercise 2
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 18 May 2023
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

def linear(fun, A, x, b, grad, tol=1e-6, maxit=50000):

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
    g = grad(A,x,b)
    grad_norm = np.linalg.norm(g)
    d = b-np.dot(A,x)
    it = 0
	
    while grad_norm>=tol and it<maxit:
        w = np.dot(A,d)
        alpha = (grad_norm**2)/(np.dot(np.transpose(d),w))
	    
        grad_normPrev = grad_norm
        x = x + alpha*d
        g = g + alpha*w
        grad_norm = np.linalg.norm(grad(A,x,b))
        B = (grad_norm**2)/(grad_normPrev**2)
        d = -g + np.dot(B,d)
        it = it + 1
        

    return x, it, grad_norm


def normal(fun, A, x, b, grad, tol=1e-6, maxit=50000):

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
    g = grad(A,x,b)
    grad_norm = np.linalg.norm(g)
    d = np.dot(np.negative(np.transpose(A)),g)
    z = np.negative(d)
    z_norm = np.linalg.norm(z)
    it = 0
	
    while grad_norm>=tol and it<maxit:
        w = np.dot(A,d)
        w_norm = np.linalg.norm(w)
        alpha = (z_norm**2)/(w_norm**2)
	    
        z_normPrev = z_norm
        
        x = x + alpha*d
        g = g + alpha*w
        z = np.dot(np.transpose(A),g)
        grad_norm = np.linalg.norm(g)
        z_norm = np.linalg.norm(z)
        B = (z_norm**2)/(z_normPrev**2)
        d = -z + np.dot(B,d)
        it = it + 1
        

    return x, it, grad_norm
