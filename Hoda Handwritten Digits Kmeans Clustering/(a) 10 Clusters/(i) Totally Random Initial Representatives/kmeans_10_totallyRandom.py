
# K-means Algocrithm


#10 clusters with 10 random 0/1 initial representatives 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def ith_image(matrix,i):
    return np.array(matrix[i,:].reshape((40,30)))
    
    

def main():
    
    with open('TrainData.txt', 'r') as f:
        dataset = np.array([[int(num) for num in line.split(',')] for line in f])
    
    
       
    init_rep = np.random.randint(2, size=(10,1200))
    kmeans = KMeans(n_clusters=10, init=init_rep, random_state=0,n_init=1).fit(dataset)
    centers = kmeans.cluster_centers_
    

    print("\n[+] Average of distances of samples to their closest cluster center:\n" + str(kmeans.inertia_ /20000))

    plt_rs = 2
    plt_cs = 5
    axes=[]
    fig_i=plt.figure(num="initial representatives")
    for k in range(0,10):
        r=ith_image(init_rep,k) 
        axes.append( fig_i.add_subplot(plt_rs, plt_cs, k+1) )
        plt.axis('off')  
        imag=plt.imshow(r, interpolation='nearest')  
        imag=plt.imshow(r)
        imag.set_cmap('gray')
    fig_i.tight_layout()
    plt.savefig('initial_representatives.png',bbox_inches='tight')    
    plt.show(block=False)
    
    axes=[]
    fig=plt.figure(num="final representatives")
    for k in range(0,centers.shape[0]):
        r=ith_image(centers,k) 
        axes.append( fig.add_subplot(plt_rs, plt_cs, k+1) )
        plt.axis('off')
        plt.title("Center:"+str(k))  
        imag=plt.imshow(r, interpolation='nearest')  
        imag=plt.imshow(r)
        imag.set_cmap('gray')
    fig.tight_layout()
    plt.savefig('final_representatives.png',bbox_inches='tight')    
    plt.show(block=False)



    with open('TestData.txt', 'r') as f:
        tests = np.array([[int(num) for num in line.split(',')] for line in f])

    test_result = kmeans.predict(tests)
    
    resr=2
    resc=7
    axes=[]
    fig=plt.figure(num="13 test data")
    for k in range(0,tests.shape[0]):
        r=ith_image(tests,k) 
        axes.append( fig.add_subplot(resr, resc, k+1) )
        plt.axis('off')  
        imag=plt.imshow(r, interpolation='nearest')  
        imag=plt.imshow(r)
        imag.set_cmap('gray')
    fig.tight_layout()
    plt.savefig('TestData.png',bbox_inches='tight')    
    plt.show(block=False)
   
    axes=[]
    fig2=plt.figure(num="tests result")
    for k in range(0,test_result.size):
        r=ith_image(centers,test_result[k]) 
        axes.append( fig2.add_subplot(resr, resc, k+1) )
        plt.axis('off')  
        plt.title("Label:"+str(test_result[k]))
        imag=plt.imshow(r, interpolation='nearest')  
        imag=plt.imshow(r)
        imag.set_cmap('gray')
    fig2.tight_layout()
    plt.savefig('test_result.png',bbox_inches='tight')    
    plt.show(block=True)


main()
