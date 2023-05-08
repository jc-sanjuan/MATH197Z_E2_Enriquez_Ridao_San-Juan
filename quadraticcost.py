"""
MATH 197 Z Exercise 1
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 16 April 2023
"""
import numpy as np
from optquad import steepest_descent

def quadraticcost(A,x,b):
    """
	Parameter
	---------
 		A:matrix
          input n dimensional matrix
             
 		x:array
 		  input n dimensional array
           
        b:array
           input n dimensional array

	Returns
	-------
		function:float
            quadratic cost function
	"""

    function = (1/2)*np.dot(np.transpose(x),np.dot(A,x))-np.transpose(b)*x
    
    return function

def grad_quad(A,x,b):
    """
	Parameter
	---------
        A:matrix
          input n dimensional matrix
            
		x:array
		  input n dimensional array
          
        b:array
          input n dimensional array
	Returns
	-------
		dx:3d vector
          gradient
	"""

    dx = np.dot(A,x)-b
    return dx


if __name__ == "__main__":
    """
    If randomized matrix:
    R = np.random.rand(3,3)
    A = np.dot(R,R)+10.*np.identity(3)
    """
    gamma = int(input("Input a value: "))
    A = np.array([[1,0,0],[0,gamma,0],[0,0,gamma**2]])
    x = np.array([[0],[0],[0]])
    b = np.array([[-1],[-1],[-1]])

    
    x, it, grad_norm = steepest_descent(quadraticcost, A, x, b, grad_quad)
    print("Approximate Minimizer: {}" .format(x))
    print("Gradient Norm 		: {}" .format(grad_norm))
    print("Number of Iterations	: {}" .format(it))
    print("Function Value		: {}" .format(quadraticcost(A,x,b)))

