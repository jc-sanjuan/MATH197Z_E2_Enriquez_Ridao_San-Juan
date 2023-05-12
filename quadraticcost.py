"""
MATH 197 Z Exercise 2
Name: Deangelo Enriquez
Raphaell Ridao
Jan Catherine San Juan
Date: 18 May 2023
"""
import numpy as np
from optquad import steepest_descent, linear, normal
import sys

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
    choice = input("Choose a method:\n 1. Steepest Descent Method\n 2. Linear Conjugate Gradient Method\n 3. Conjugate Gradient Normal Residual Method\n 4. Conjugate Residual Method\n Input name/number/acronym: ")
    gamma = int(input("Input a numerical value: "))
    
    A = np.array([[1,0,0],[0,gamma,0],[0,0,gamma**2]])
    x = np.array([[0],[0],[0]])
    b = (-1)*np.array([[1],[1],[1]])

    K = gamma**2
    
    if choice == 'Steepest Descent Method' or choice == '1':
        x, it, grad_norm = steepest_descent(quadraticcost, A, x, b, grad_quad)
    elif choice == 'Linear Conjugate Gradient Method' or choice == '2':
        x, it, grad_norm = linear(quadraticcost, A, x, b, grad_quad)
    elif choice == 'Conjugate Gradient Normal Residual Method' or choice == '3':
        x, it, grad_norm = normal(quadraticcost, A, x, b, grad_quad)
    elif choice == 'Conjugate Residual Method' or choice == '4':
        x, it, grad_norm = normal(quadraticcost, A, x, b, grad_quad)#change function name if meron na
    else:
        print("Please input a valid number or the exact method name.")
        sys.exit()
        

    print("Approximate Minimizer: {}" .format(x))
    print("Gradient Norm 		: {}" .format(grad_norm))
    print("Number of Iterations	: {}" .format(it))
    print("Function Value		: {}" .format(quadraticcost(A,x,b)))
    print("K(A)		: {}" .format(K))
