###############################################################################################
#       Computing Inverse Of An Invertible Square Matrix By The Use Of QR decomposition       #               
#                                                                			      #
#                                                                                             #
#       [RunGuide]:                                                                           #
#            write input matrix in a file named 'testCase.txt' and then RUN:                  #
#            python[version] Inverse.py                                                       #
#                                                                                             #
###############################################################################################

import numpy as np


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


def InverseMatrix(A):

    (rows, cols) = A.shape
    inverse = np.zeros((rows, cols))
    (Q,R)=np.array(EQR_Factorization(A))
    Q_transpose = Q.transpose()
    
    
    for inverseMatrixColumnNo in range(0,rows):

        b = Q_transpose[:,inverseMatrixColumnNo]
        x_vector = np.zeros(rows)


        for i in range (rows-1 , -1,-1):  
            tmp=b[i]
            for j in range(rows-1,i,-1):
                tmp -= R[i,j]*x_vector[j]

            x_vector[i] = tmp / R[i,i]

        inverse[:,inverseMatrixColumnNo] = x_vector
            
    return inverse







def main():
    with open('testCase.txt', 'r') as f:
        matrix = np.array([[float(num) for num in line.split(',')] for line in f])
    
    Inverse=InverseMatrix(matrix)


    #Exact Numbers:
    # print('\n[+] Matrix :\n')
    # print(np.round(matrix,2))
    # print('\n[+] Inverse:\n')
    # print( np.round(Inverse,2))
    # print('\n[+] Matrix * Inverse :\n')
    # print(np.matmul(matrix,Inverse))

    #Rounded Nembers
    print('\n[+] Matrix :\n')
    print(matrix)
    print('\n[+] Inverse:\n')
    print( Inverse)
    print('\n[+] Matrix * Inverse : (Must Be I)\n')
    print(np.round(np.matmul(matrix,Inverse)))

 

main()