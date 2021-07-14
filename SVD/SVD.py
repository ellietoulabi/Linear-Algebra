
# Prerequirements :
#       [+] sudo apt-get install python3-tk
#       [+] python -m pip install  matplotlib
#       [+] python -m pip install  numpy


import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy.linalg as la


if __name__ == "__main__":

    Image = img.imread('lena.png') # 512x512 image

    r =np.linalg.matrix_rank(Image)
       
    U, S, V = np.linalg.svd(Image, full_matrices=False)

    ordered_S = S
    ordered_S[::-1].sort()
  
    plt.plot(ordered_S)
    plt.title('Ordered Singular Values')
    plt.ylabel('singular values')
    plt.xlabel('index of singular values in descending order')
    plt.savefig('SVD-singular values.png')
    plt.show(block=False)

   
    
    U_r = U[:,:r]
    V_r = V[:r,:]
    S_r = S[:r]

    

    
    comps = [r,10, 50, 100, 150, 200]
    
    fig=plt.figure(figsize = (16, 8),num='Compression Results')
    for i in range(6):
        result = U_r[:, :comps[i]] @ np.diag(S_r[:comps[i]]) @ V_r[:comps[i], :]
        if(i  == 0):
            plt.subplot(2, 3, i+1), plt.imshow(result, cmap = 'gray',interpolation='nearest'), plt.axis('off'), plt.title("Original p = " + str(comps[i]))
            
        else:
            plt.subplot(2, 3, i+1), plt.imshow(result, cmap = 'gray',interpolation='nearest'), plt.axis('off'), plt.title("p =" + str(comps[i]))
    plt.savefig('SVD-Compression results.png') 
    fig.tight_layout()      
    plt.show()


   
