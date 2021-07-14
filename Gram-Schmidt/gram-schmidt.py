
# Gram-Schmidt Algorithm Implementation

import numpy as np
import math as mt


def gram_schmidt(A):

    (rows, cols) = A.shape
    basis = np.zeros((rows, cols))
    dependents = np.zeros(cols)
    dependents_coefs = np.zeros((rows,cols))
    i_cnt = 0
    d_cnt = 0

    for j in range(0, cols):
        cur_vector = A[:,j]
        keep_vector = A[:,j]
        for i in range(0,j):    
            inner_p = np.inner(cur_vector, basis[:,i])
            cur_vector = cur_vector - ((inner_p)*basis[:,i])
        
        norm = np.sqrt(np.inner(cur_vector,cur_vector))
        if(norm > 10E-5 ):
            cur_vector = cur_vector / norm
            basis[:,i_cnt] = cur_vector
            i_cnt = i_cnt + 1
        else:
            dependents[d_cnt] = j
            for i in range(0,i_cnt):
                dependents_coefs[i,d_cnt] = np.inner(keep_vector, basis[:,i])
            d_cnt = d_cnt + 1

    return basis[:,:i_cnt], dependents[:d_cnt], dependents_coefs[:i_cnt,:d_cnt]

def main():
    with open('testCase.txt', 'r') as f:
        matrix = np.array([[int(num) for num in line.split(',')] for line in f])
    
    (basis, dependents, dependents_coefs)=gram_schmidt(matrix)
    print('\n[+] Basis of Vector Space:')
    print('[-] each column is a vector\n')
    print(basis)
    print('\n[+] Linearly Dependent Vectors:')
    print('[-] Each row is index of linearly dependent vector(column number) in input matrix\n')
    print(dependents)
    print('\n[+] Coefficients Of Linearly Dependent Vectors:')
    print('[-] [i][j]th element is coefficient of j-th dependent vector for ic-th basis vector\n')
    print(dependents_coefs)

main()