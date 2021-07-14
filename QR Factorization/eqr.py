
# QR Factorization Implementation 

import numpy as np
import math as mt


def EQR_Factorization(A):

    (rows, cols) = A.shape
    Q_matrix = np.zeros((rows, cols))
    R_matrix = np.zeros((cols, cols))
    dependents = np.zeros(cols)
    i_cnt = 0
    d_cnt = 0
    
    for j in range(0, cols):
        cur_vector = A[:,j]
        keep_vector = A[:,j]
        dep = np.zeros((cols, 1))
        
        for i in range(0,j):    
            inner_p = np.inner(cur_vector, Q_matrix[:,i])
            dep[i,0] = inner_p
            cur_vector = cur_vector - ((inner_p)*Q_matrix[:,i])
            
        norm = np.sqrt(np.inner(cur_vector,cur_vector))
        
        if(norm > 10E-5 ):
            dep[i_cnt,0] = norm
            R_matrix[0:cols,j]=dep[0:cols,0]
            cur_vector = cur_vector / norm
            Q_matrix[:,i_cnt] = cur_vector
            i_cnt = i_cnt + 1
        else:
            dependents[d_cnt] = j
            for i in range(0,i_cnt):
                R_matrix [i,j]= np.inner(keep_vector, Q_matrix[:,i])
            d_cnt = d_cnt + 1


    return Q_matrix[:,:i_cnt], R_matrix[:i_cnt,:]

def main():
    with open('testCase.txt', 'r') as f:
        matrix = np.array([[int(num) for num in line.split(',')] for line in f])
    
    (Q_matrix, R_matrix)=EQR_Factorization(matrix)
    print('\n[+] Q Matrix :\n')
    print(np.round(Q_matrix,2))
    print('\n[+] R Matrix :\n')
    print( np.round(R_matrix,2))
    print('\n[+] Q R :\n')
    print(np.matmul(Q_matrix,R_matrix))

 

main()